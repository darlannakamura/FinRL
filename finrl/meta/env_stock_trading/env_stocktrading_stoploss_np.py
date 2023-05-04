from __future__ import annotations

import random

import gym
import numpy as np
from numpy import random as rd
from stable_baselines3.common.logger import Logger, make_output_format

logger = Logger("logs", output_formats=[make_output_format("csv", "logs")])


class StockTradingEnvStopLoss(gym.Env):
    def __init__(
        self,
        config,
        initial_account=1e6,
        gamma=0.99,
        turbulence_thresh=99,
        min_stock_rate=0.1,
        max_stock=1e2,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=2**-11,
        initial_stocks=None,
        hmax=10,
        stoploss_penalty=0.9,
        profit_loss_ratio=2,
        cash_penalty_proportion=0.1,
        daily_information_cols=["open", "close", "high", "low", "volume"],
        shares_increment=1,
        discrete_actions=False,
        random_start=True,
        patient=False
    ):
        price_ary = config["price_array"]
        ohlcv_ary = config["ohlcv_array"]
        tech_ary = config["tech_array"]
        turbulence_ary = config["turbulence_array"]
        if_train = config["if_train"]
        self.price_ary = price_ary.astype(np.float32)
        self.ohlcv_ary = ohlcv_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary
        self.turbulence_threshold = turbulence_thresh
        self.random_start = random_start
        self.patient = patient

        self.tech_ary = self.tech_ary * 2**-7
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (
            self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2**-5
        ).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "StockTradingEnvStopLoss"
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        # self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]

        self.daily_information_cols = daily_information_cols

        self.state_dim = (
            1 + stock_dim +  self.price_ary.shape[1] + self.tech_ary.shape[1]
        )

        self.stock_dim = stock_dim

        # state_dim -> why 1 + 2 ?
        # 3 * stock_dim because we can hold, buy or sell any stocks and the technical indicators can have influence on this

        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0
        self.starting_point = 0
        self.episode = -1  # initialize so we can call reset
        self.episode_history = []
        self.printed_header = False

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

        self.hmax = hmax
        self.stoploss_penalty = stoploss_penalty
        self.profit_loss_ratio = profit_loss_ratio
        self.cash_penalty_proportion = cash_penalty_proportion
        self.shares_increment = shares_increment
        self.discrete_actions = discrete_actions
        self.min_profit_penalty = 1 + profit_loss_ratio * (1 - self.stoploss_penalty)

        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []


    def reset(self):
        self.sum_trades = 0
        self.actual_num_trades = 0
        self.closing_diff_avg_buy = np.zeros(self.stock_dim)
        self.profit_sell_diff_avg_buy = np.zeros(self.stock_dim)
        self.n_buys = np.zeros(self.stock_dim)
        self.avg_buy_price = np.zeros(self.stock_dim)
        if self.random_start:
            starting_point = random.choice(range(int(self.price_ary.shape[0] * 0.5)))
            self.starting_point = starting_point
        else:
            self.starting_point = 0

        self.episode += 1
        self.day = starting_point
        price = self.price_ary[self.day]

        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0

        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        self.account_information = {
            "cash": [],
            "asset_value": [],
            "total_assets": [],
            "reward": [],
        }

        # init_state = np.array(
        #     [self.initial_capital]
        #     + [0] * self.stock_dim
        #     + self.price_ary[self.day]
        # )

        init_state = np.hstack([self.initial_capital, [0] * self.stock_dim, self.price_ary[self.day], self.tech_ary[self.day]])
        self.state_memory.append(init_state)

        return init_state


    @property
    def current_step(self):
        return self.day - self.starting_point


    def get_reward(self):
        if self.current_step == 0:
            return 0
        else:
            total_assets = self.account_information["total_assets"][-1]
            cash = self.account_information["cash"][-1]
            holdings = self.state_memory[-1][1 : self.stock_dim + 1]
            neg_closing_diff_avg_buy = np.clip(self.closing_diff_avg_buy, -np.inf, 0)
            neg_profit_sell_diff_avg_buy = np.clip(
                self.profit_sell_diff_avg_buy, -np.inf, 0
            )
            pos_profit_sell_diff_avg_buy = np.clip(
                self.profit_sell_diff_avg_buy, 0, np.inf
            )

            cash_penalty = max(0, (total_assets * self.cash_penalty_proportion - cash))
            if self.current_step > 1:
                prev_holdings = self.state_memory[-2][1 : self.stock_dim + 1]
                stop_loss_penalty = -1 * np.dot(
                    np.array(prev_holdings), neg_closing_diff_avg_buy
                )
            else:
                stop_loss_penalty = 0
            low_profit_penalty = -1 * np.dot(
                np.array(holdings), neg_profit_sell_diff_avg_buy
            )
            total_penalty = cash_penalty + stop_loss_penalty + low_profit_penalty

            additional_reward = np.dot(np.array(holdings), pos_profit_sell_diff_avg_buy)

            reward = (
                (total_assets - total_penalty + additional_reward) / self.initial_capital
            ) - 1
            reward /= self.current_step

            return reward

    def step(self, actions):
        if self.printed_header is False:
            self.log_header()

        # let's just log what we're doing in terms of max actions at each step.
        self.sum_trades += np.sum(np.abs(actions))

        if self.day == self.price_ary.shape[0] - 1:
            # if we hit the end, set reward to total gains (or losses)
            return self.return_terminal(reward=self.get_reward())

        # compute value of cash + assets
        begin_cash = self.state_memory[-1][0]
        holdings = self.state_memory[-1][1 : self.stock_dim + 1]
        assert min(holdings) >= 0
        closings = self.price_ary[self.day]
        asset_value = np.dot(holdings, closings)
        
        reward = self.get_reward()

        self.account_information["cash"].append(begin_cash)
        self.account_information["asset_value"].append(asset_value)
        self.account_information["total_assets"].append(begin_cash + asset_value)
        self.account_information["reward"].append(reward)

        # multiply action values by our scalar multiplier and save
        actions = actions * self.hmax
        self.actions_memory.append(
            actions * closings
        )  # capture what the model's trying to do
        # buy/sell only if the price is > 0 (no missing data in this particular date)
        actions = np.where(closings > 0, actions, 0)

        if self.turbulence_threshold is not None:
            # if turbulence goes over threshold, just clear out all positions
            if self.turbulence_bool[self.day] != 0:
                actions = -(np.array(holdings) * closings)
                self.log_step(reason="TURBULENCE")
        # scale cash purchases to asset
        if self.discrete_actions:
            # convert into integer because we can't buy fraction of shares
            actions = np.where(closings > 0, actions // closings, 0)
            actions = actions.astype(int)
            # round down actions to the nearest multiplies of shares_increment
            actions = np.where(
                actions >= 0,
                (actions // self.shares_increment) * self.shares_increment,
                ((actions + self.shares_increment) // self.shares_increment)
                * self.shares_increment,
            )
        else:
            actions = np.where(closings > 0, actions / closings, 0)

        # clip actions so we can't sell more assets than we hold
        actions = np.maximum(actions, -np.array(holdings))

        self.closing_diff_avg_buy = closings - (
            self.stoploss_penalty * self.avg_buy_price
        )
        if begin_cash >= self.stoploss_penalty * self.initial_capital:
            # clear out position if stop-loss criteria is met
            actions = np.where(
                self.closing_diff_avg_buy < 0, -np.array(holdings), actions
            )

            if any(np.clip(self.closing_diff_avg_buy, -np.inf, 0) < 0):
                self.log_step(reason="STOP LOSS")

        # compute our proceeds from sells, and add to cash
        sells = -np.clip(actions, -np.inf, 0)
        proceeds = np.dot(sells, closings)
        costs = proceeds * self.sell_cost_pct
        coh = begin_cash + proceeds
        # compute the cost of our buys
        buys = np.clip(actions, 0, np.inf)
        spend = np.dot(buys, closings)
        costs += spend * self.buy_cost_pct
        # if we run out of cash...
        if (spend + costs) > coh:
            if self.patient:
                # ... just don't buy anything until we got additional cash
                self.log_step(reason="CASH SHORTAGE")
                actions = np.where(actions > 0, 0, actions)
                spend = 0
                costs = 0
            else:
                # ... end the cycle and penalize
                return self.return_terminal(
                    reason="CASH SHORTAGE", reward=self.get_reward()
                )

        self.transaction_memory.append(actions)  # capture what the model's could do

        # get profitable sell actions
        sell_closing_price = np.where(
            sells > 0, closings, 0
        )  # get closing price of assets that we sold
        profit_sell = np.where(
            sell_closing_price - self.avg_buy_price > 0, 1, 0
        )  # mark the one which is profitable

        self.profit_sell_diff_avg_buy = np.where(
            profit_sell == 1,
            closings - (self.min_profit_penalty * self.avg_buy_price),
            0,
        )

        if any(np.clip(self.profit_sell_diff_avg_buy, -np.inf, 0) < 0):
            self.log_step(reason="LOW PROFIT")
        else:
            if any(np.clip(self.profit_sell_diff_avg_buy, 0, np.inf) > 0):
                self.log_step(reason="HIGH PROFIT")

        # verify we didn't do anything impossible here
        assert (spend + costs) <= coh

        # log actual total trades we did up to current step
        self.actual_num_trades = np.sum(np.abs(np.sign(actions)))

        # update our holdings
        coh = coh - spend - costs
        holdings_updated = holdings + actions

        # Update average buy price
        buys = np.sign(buys)
        self.n_buys += buys
        self.avg_buy_price = np.where(
            buys > 0,
            self.avg_buy_price + ((closings - self.avg_buy_price) / self.n_buys),
            self.avg_buy_price,
        )  # incremental average

        # set as zero when we don't have any holdings anymore
        self.n_buys = np.where(holdings_updated > 0, self.n_buys, 0)
        self.avg_buy_price = np.where(holdings_updated > 0, self.avg_buy_price, 0)

        self.day += 1

        # Update State
        state = np.hstack([coh, list(holdings_updated), self.price_ary[self.day], self.tech_ary[self.day]])
        self.state_memory.append(state)

        return state, reward, False, {}

    def log_header(self):
        self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"  # column widths: 8, 10, 15, 7, 10
        print(
            self.template.format(
                "EPISODE",
                "STEPS",
                "TERMINAL_REASON",
                "CASH",
                "TOT_ASSETS",
                "TERMINAL_REWARD_unsc",
                "GAINLOSS_PCT",
                "CASH_PROPORTION",
            )
        )
        self.printed_header = True

    def log_step(self, reason, terminal_reward=None):
        if terminal_reward is None:
            terminal_reward = self.account_information["reward"][-1]
        cash_pct = (
            self.account_information["cash"][-1]
            / self.account_information["total_assets"][-1]
        )
        gl_pct = self.account_information["total_assets"][-1] / self.initial_capital
        rec = [
            self.episode,
            self.day - self.starting_point,
            reason,
            f"${'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}",
            f"${'{:0,.0f}'.format(float(self.account_information['total_assets'][-1]))}",
            f"{terminal_reward*100:0.5f}%",
            f"{(gl_pct - 1)*100:0.5f}%",
            f"{cash_pct*100:0.2f}%",
        ]
        self.episode_history.append(rec)
        print(self.template.format(*rec))

    def return_terminal(self, reason="Last Date", reward=0):
        state = self.state_memory[-1]
        self.log_step(reason=reason, terminal_reward=reward)
        # Add outputs to logger interface
        gl_pct = self.account_information["total_assets"][-1] / self.initial_capital
        logger.record("environment/GainLoss_pct", (gl_pct - 1) * 100)
        logger.record(
            "environment/total_assets",
            int(self.account_information["total_assets"][-1]),
        )
        reward_pct = self.account_information["total_assets"][-1] / self.initial_capital
        logger.record("environment/total_reward_pct", (reward_pct - 1) * 100)
        logger.record("environment/total_trades", self.sum_trades)
        logger.record(
            "environment/actual_num_trades",
            self.actual_num_trades,
        )
        logger.record(
            "environment/avg_daily_trades",
            self.sum_trades / (self.current_step),
        )
        logger.record(
            "environment/avg_daily_trades_per_asset",
            self.sum_trades / (self.current_step) / self.stock_dim,
        )
        logger.record("environment/completed_steps", self.current_step)
        logger.record(
            "environment/sum_rewards", np.sum(self.account_information["reward"])
        )
        logger.record(
            "environment/cash_proportion",
            self.account_information["cash"][-1]
            / self.account_information["total_assets"][-1],
        )
        return state, reward, True, {}

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
