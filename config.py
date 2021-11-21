class config:
    def __init__(self) -> None:
        self.width = 8
        self.height = 8
        self.n_in_row = 5

        # MCTS
        self.init_model = False
        # self.init_model = "data/models/best_policy.model"
        self.n_games = 5
        self.use_gpu = True
        self.play_batch_size = 10
