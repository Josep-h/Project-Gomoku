class config:
    def __init__(self) -> None:
        self.width = 11
        self.height = 11
        self.n_in_row = 5
        self.depth = 4
        self.total_games = 5

        # MCTS
        self.init_model = False
        # self.init_model = "data/models/best_policy.model"
        self.init_model = "data/models/current_policy_549.model"
        self.n_games = 10
        self.use_gpu = False
        self.play_batch_size = 1
        self.play_out_num = 20
        self.check_freq = 50

        # Compare
        self.show = False
