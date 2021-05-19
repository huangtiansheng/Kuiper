# Minor Comments from Tiansheng
The entrance of the program is ``training/evals/manager.py``. The basic logic of this file is as belows:
1. Step into the `process_cmd(.)` function.
2. Load the configuration from designated yaml file, and do all kinds of initialization (e.g. ip address of the physical machines, and their GPU information)
3. Run `training/param_server.py` on the master node, and `training/learner.py` on worker node. Note: Each worker node sets up a specific learner process for each GPU.  
4. Use `subprocess.Popen(.)` to achieve logging.

## Param_server Process
Then we move on to ``training/param_server.py`` to see what happen. We first see the main function of this file.
1. Initialize random seed, manager for multi-processing, and register channels (queues essentially) for communication between processes. 
2. Initialize dataset (train set and test set), and the corresponding neural model (e.g. resnet).
3. Step into the ``init_myprocesses(.)`` function

Let's see what happen in ``init_myprocesses(.)``:
1. Init process group to achieve data parallel.
2. Step into `initiate_sampler_query(.)` to initiate the clientSampler (the core part of participation selection)
3. After initialization of sampler, perform the following steps:  
   3.1 For each worker process, use the initiated `clientSampler` to obtain `nextClientIdToRun` (which should only contain one client_id).  
   3.2 Store `nextClientIdToRun` into a list ``clientIdsToRun``, whose index corresponds to the workers index.  
   3.3 Then broadcast  ``clientIdsToRun`` to the worker processes.  
   3.4 I personally do not fully understand why we need to schedule a client for each worker process. That is, the real intention of doing `clientSampler.nextClientIdToRun(hostId=wrank)`  
4. Finally, `init_myprocesses(.)` ends, and then the server process formally goes into the run stage via `fn(.)` (which is in fact `run()`).

Now we look at  ``run(.)`` to see what's the next:
1. Broadcast the global model weights to all the learners.
2. Do all kinds of initialization (skip for discussion for now).
3. Then it steps into the "while loop" in line 203, with two break condition: 1. timeout of simulation time 2.exceed the pre-set epoch 

Now we look into the "while loop" in line 203:
1. Get the statistics (e.g. loss, speed, etc) for previous rounds of training. This is done
   by fetching from the queue via ``tmp_dict = queue.get()``
2. Step into the "for loop" to iterate clients from a specific learner.  
   2.1. Apply the update from selected clients into `sumDeltaWeights`  
   2.2. Use  `clientSampler.registerScore()` to update the client utility using `iteration_loss`(i.e., empirical loss, to achieve statistics utility) and `virtualClientClock`(i.e., duration, to achieve system utility)  
   2.3. Exit the "For loop" after all the clients of a learner has been iterated.
3. Aggregate the test results of different clients run in this leaner process, and visualize the results.
4. Some operations based on "staleness". This staleness definition is quite confusing. I don't see why the process does not need a "waiting before synchronization" step. That is, do model aggregation and all kinds of synchronized jobs only when training results from all the clients received from queue? 
   So, let's skip this part to line 371, after which the synchronized process starts.
5. The first thing to do before aggregation is to re-sample the clients using ``clientSampler.resampleClients(numToSample, cur_time=epoch_count))``.
   This function yields a set of selected clients  in the next iteration.
6. Then remove the "dummy clients"  that we are not going to run via  function ``prune_client_tasks(.)``. (Let's skip this)
7. After removal, allocate the clients to worker process and do all kinds of related update.
8. Update the global model parameter via  ``param.data += sumDeltaWeights[idx]``.
9. Broadcast global parameters (i.e., `param.data`), selected clients(i.e., `clientsList`), and their allocation on workers (i.e., `clientIdsToRun`) to all the worker processes. 


## Learner/worker Process
Besides, let's see what happen in the learner process (see `training/learner.py`)
1. Initialize seed channel and connect to the Param_server process.
2. Init `train_dataset`,`test_dataset` (note: they are the intact datasets)
3. Construct the data partitioner `global_trainDB` and do partitioning on `train_dataset`, i.e., split the intact train_dataset into # of `args.total_worker` parts, which is stored in
`global_trainDB`. And same partitioning is done by `global_testDB`. Both `global_trainDB` and `global_testDB` are global variables in the learner process. 
4. Step 4 in the learner/worker process corresponds to step 3 in the `init_myprocesses()` of the server process. It basically covers the following sub-steps:  
   4.1. Step into `report_data_info(.)` to indicate data information. Specifically, this function transmits data distance and size of partitions to server via `queue`.  
   4.2. And then,receive the broadcasted `clientIdToRun` from the server.  
   4.3. Derive the client_id scheduled for this worker process via ``nextClientIds=[clientIdToRun[args.this_rank - 1].item()]``  
   4.4. And note, `nextClientIds` is a global variable, and will be used later. 
5. Delete `train_dataset`,`test_dataset` after construction of partitioners (since they are no longer useful, all the training is done by partitioner).
6. Step into `init_myprocesses(.)`, and subsequently steps into the `run(.)` to formally start running.

Let's see what happen in ``run(.)``:
1. Receive the global model weights from server process via ``dist.broadcast(tensor=param.data, src=0)``
2. Store the updated weights into ``last_model_tensors`` (which is a list recording model parameters of all the iterations)
3. Do all the initialization until the "for loop" in line 457, which loops over ``args.epochs`` number of epochs (or iterations)

Then we look into the first "for loop" in line 457 :
1. Decay the learning_rate.
2. Step into the second "for loop" over `nextClientIds` (which indicates the scheduled clients on this worker process).
   But still, I cannot understand why the server only choose one client for each worker process in the first round 
   (see step 3 of `init_myprocesses(.)` of the server process)  
   2.1. Then, this learner process run the function ``run_client(.)`` for each scheduled clients on this learner process.
   This could be understood as the real training of a virtual client (note, a learner process might run several clients' training process in each iteration)
   For now, we first skip the detailed inspect on ``run_client(.)``, and this content will be presented later.   
   2.2. Record all the statistics for each of the  clients who have been involved (e.g., `trainedModels`,`preTrainedLoss`, `trainSpeed`)After the second "for loop" (i.e., training of all allocated clients finished), do the following steps:

After the second "for loop" (i.e., training of all allocated clients finished), do the following steps:
1. Put the statistics into ``queue`` towards the server.
2. Record the sendDur (real time duration for sending)
3. Synchronous parameters from the server via `dist.broadcast(tensor=tmp_tensor, src=0)` and add it to the list `last_model_tensors`
4. Receive the next `client_tensor`  and `step_tensor` from the server. `client_tensor` is a list that captures the sampled clients_id
, and `step_tensor` determines the exact allocation of clients for each learner process.
5. Test the model accuracy if needed.
6. If the queue stop_flag contains element (which is issued by the server), break the first "for loop" and exit.

Now we check how the function ``run_client(.)`` works:
1. Initialize `optimizer`, `criterion` (loss template).
2. Derive the client's data ``client_train_data`` from data partitioner `global_trainDB` via function `select_dataset(.)`
3. Step into the "for loop" in line 216 to do multiple iterations of SGD  
   3.1. Derive (data,target) from the client's train data  
   3.2. Flip the label(target) if the client is malicious.  
   3.3. Calculate Loss based on criterion and (data,target).
   3.4. Backward using optimizer.
4. Record the statistics (e.g., training speed, training loss) of this client, and return to learner process. 

##Overall Comments
The complexity of Kuiper is hard to say low, partly because it involves parallel training based on multiple processes.
Despite the implementation of parallel training, I personally insist that Kuiper is still essentially a simulation platform, but
not a real usable system for federated learning, since it still requires each process to simulate the training of multiple clients.  

The core part of Kuiper (i.e., its selection mechanism) is easy to grasp but only takes a small portion of code space, let me wondering
the focus of this study, either it is to build a usable simulation platform, or to implement their client selection policy. 
In view of the simulation platform itself (which implements parallel computing on multiple GPUs), I think the design can be mimicked, but is not so successful considering its weak extensibility as a simulation platform.