This is a lite version of our code and will be updated in the future alongside the camera ready version of the paper.

The included code should be sufficient to run experiments on RC. In the code this is referred to as the treasure_hunt game.
It is the responsibility of the user to install required dependencies. Namely, we require the Sacred (https://github.com/IDSIA/sacred) library.

# Instructions
We recommend running experiments on medium or small (trivial) sized games with a learning rate of 1e-4 to begin. These should take several hours to give reasonable performance.

For medium sized games:
>> python experiments/treasure_hunt_gpu.py with GPU 'map_seed=4' 'learner_learn_from_ground_truth=False' 'learner_lr=1e-4' 'learner_batch_size=128'  medium  monotonic=True medium_model UNIF_STATE

For the same grid size as the paper:
>> python experiments/treasure_hunt_gpu.py with GPU 'map_seed=4' 'learner_learn_from_ground_truth=False' 'learner_lr=1e-4' 'learner_batch_size=128'  big  monotonic=True medium_model UNIF_STATE

Training on CPU should be doable (not tested much) but is expected to be slow due to our implementation of upper-concave-envelopes (see appendix for details).

# Results
Results are saved in the `treasure_hunt_results` folder, and include EPFs (predicted, one step lookahead, and true), at the root and the worst 16 states, as well as histograms of the losses.
Convergence can be somewhat slow and we encourage the user to run the experiment for at least 1M iterations.
