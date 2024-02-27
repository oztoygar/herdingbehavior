import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from collections import deque
from tqdm import tqdm
import networkx as nx

random.seed(123456)
np.random.seed(123456)

import heapq

class OrderBook:
    def __init__(self):
        self.buy_orders = []
        self.sell_orders = []
        self.trade_history = []
        self.order_id_counter = 0
        self.orders = {}
        # Initialize total liquidity for buy and sell orders
        self.total_buy_liquidity = 0
        self.total_sell_liquidity = 0

    def add_order(self, order):
        order_id = self.order_id_counter
        self.order_id_counter += 1
        self.orders[order_id] = order
        order['matched'] = False  # Add a 'matched' flag to the order

        # Adjust total liquidity based on order type
        if order['type'] == 'buy':
            heapq.heappush(self.buy_orders, (-order['price'], order['timestamp'], order['quantity'], order_id, order['agentid']))
            self.total_buy_liquidity += order['quantity']
        else:
            heapq.heappush(self.sell_orders, (order['price'], order['timestamp'], order['quantity'], order_id, order['agentid']))
            self.total_sell_liquidity += order['quantity']


    def match_orders(self, herding_index):
        executed_trades = []  # List to store details of executed trades
        while self.buy_orders and self.sell_orders and -self.buy_orders[0][0] >= self.sell_orders[0][0]:
            buy_order = heapq.heappop(self.buy_orders)  # Highest buy order
            sell_order = heapq.heappop(self.sell_orders)  # Lowest sell order
            execution_quantity = min(buy_order[2], sell_order[2])

            if execution_quantity <= 0:
                continue  # Skip if no trade is possible

            # New execution price logic: midpoint adjusted for market impact
            midpoint_price = (-buy_order[0] + sell_order[0]) / 2
            market_impact = self.calculate_market_impact(execution_quantity, herding_index)
            execution_price = midpoint_price * (1 + market_impact)

            self.orders[buy_order[3]]['matched'] = True
            self.orders[sell_order[3]]['matched'] = True

            executed_trades.append({
                'buy_order_id': buy_order[3],
                'sell_order_id': sell_order[3],
                "buyer_id": buy_order[4],
                "seller_id": sell_order[4],
                'price': execution_price,
                'quantity': execution_quantity,
                'timestamp': market.current_round
            })

            self.trade_history.append({'buyer_id':buy_order[4], 'seller_id': sell_order[4],'buy_order_id': buy_order[3], 'sell_order_id': sell_order[3], 'price': execution_price, 'quantity': execution_quantity, 'timestamp': market.current_round})

            # Update liquidity and re-push partially executed orders
            self.update_liquidity_and_repush_orders(buy_order, sell_order, execution_quantity)

        return executed_trades

    def update_liquidity_and_repush_orders(self, buy_order, sell_order, execution_quantity):
        # Adjust total liquidity
        self.total_buy_liquidity -= execution_quantity
        self.total_sell_liquidity -= execution_quantity

        # Check for partial orders and re-push them with updated quantities
        if buy_order[2] > execution_quantity:
            new_buy_order = (buy_order[0], buy_order[1], buy_order[2] - execution_quantity, buy_order[3], buy_order[4])
            heapq.heappush(self.buy_orders, new_buy_order)
        if sell_order[2] > execution_quantity:
            new_sell_order = (sell_order[0], sell_order[1], sell_order[2] - execution_quantity, sell_order[3], sell_order[4])
            heapq.heappush(self.sell_orders, new_sell_order)


    def calculate_market_impact(self, trade_quantity, herding_index):
        base_liquidity = self.total_buy_liquidity + self.total_sell_liquidity
        adjusted_liquidity = base_liquidity * (1 - herding_index)  # Reduce liquidity based on herding
        impact = (trade_quantity / adjusted_liquidity) ** (1 + herding_index)  # Non-linear impact
        return min(impact, 0.1)  # Limiting impact to a maximum of 10%

    def get_trade_history(self):
        return self.trade_history
    
    def save_trade_history_to_excel(self, filepath='trade_history.xlsx'):
        """Saves the trade history to an Excel file."""
        if self.trade_history:  # Check if there is any trade history to save
            df_trade_history = pd.DataFrame(self.trade_history)
            df_trade_history.to_excel(filepath, index=False)
        else:
            print("No trade history to save.")

    def save_orders_to_excel(self, filepath='orders.xlsx'):
        order_data = {
            'Agent ID': [],
            'Order ID': [],
            'Type': [],
            'Price': [],
            'Timestamp': [],
            'Quantity': [],
            'Matched': []  # Add Matched column
        }
        for order_id, order in self.orders.items():
            order_data['Agent ID'].append(order['agentid'])
            order_data['Order ID'].append(order_id)
            order_data['Type'].append(order['type'])
            order_data['Price'].append(order['price'])
            order_data['Timestamp'].append(order['timestamp'])
            order_data['Quantity'].append(order['quantity'])
            order_data['Matched'].append(order['matched'])  # Append matched status
        
        df = pd.DataFrame(order_data)
        df.to_excel(filepath, index=False)


    



class Agent:
    def __init__(self, id, risk_tolerance, experience, strategy, initial_wealth, initial_coins, initial_market_price):
        self.id = id
        self.risk_tolerance = risk_tolerance
        self.experience = experience
        self.strategy = strategy
        self.wealth = initial_wealth
        self.initial_wealth = initial_wealth  # Store initial wealth
        self.coin_holdings = initial_coins
        self.total_wealth = initial_wealth + (initial_coins * initial_market_price)
        self.decision = "hold"
        self.peers = None
        self.market_price = initial_market_price
        self.performance_score = 0
        self.reserved_wealth = 0
        self.reserved_coins = 0

    def calculate_final_wealth(self, final_market_price):
        # Calculate the final wealth based on coin holdings and final market price
        self.final_wealth = self.wealth + (self.coin_holdings * final_market_price)
        return self.final_wealth


    def prospect_valuation_adjustment(self, market_price):
    # Calculate the change in wealth due to changes in market price
        wealth_change = (self.coin_holdings * market_price) - (self.wealth + self.coin_holdings * self.market_price)
        
        # Parameters for prospect theory valuation
        gain_coefficient = self.risk_tolerance  # Coefficient for gains, reflecting risk tolerance
        loss_aversion_coefficient = 2.25 
        
        # If wealth_change is positive (gain), apply the gain coefficient
        if wealth_change >= 0:
            return wealth_change ** gain_coefficient
        else:
            # For losses, apply the loss aversion coefficient
            # Inverse of the risk tolerance to reflect loss aversion
            loss_coefficient = 1 / self.risk_tolerance
            # Apply loss aversion, ensuring the result is negative (indicating a loss)
            return -((-wealth_change) ** min(loss_coefficient, loss_aversion_coefficient))

    def make_decision(self, market_price, market_trend, market_news_impact, market_volatility):
        self.market_price = market_price  # Update agent's view of the market price
        overconfidence_factor = 1 + (self.experience / 10)
        if self.strategy == 'conservative':
            self.risk_tolerance = min(max(self.risk_tolerance * overconfidence_factor, 0.1), 0.5)
        elif self.strategy == 'aggressive':
            self.risk_tolerance = min(max(self.risk_tolerance * overconfidence_factor, 0.5), 0.9)
        elif self.strategy == 'balanced':
            self.risk_tolerance = min(max(self.risk_tolerance * overconfidence_factor, 0.3), 0.7)
        else:  # Adaptive strategy
            self.risk_tolerance = min(max(self.risk_tolerance * overconfidence_factor, 0.1), 0.9)
        prospect_adjustment = self.prospect_valuation_adjustment(market_price)
        self.prospect_valuation = (self.coin_holdings * market_price) + prospect_adjustment

        peer_influence = self.get_peer_influence()
        trend_influence = random.random() < (self.risk_tolerance * market_volatility) * (self.experience / 10)
        news_influence = self.evaluate_news(market_news_impact)
        influences = trend_influence + news_influence + peer_influence

        if influences > 1.5:
            self.decision = 'buy' if market_trend == 'bullish' else 'sell'
        else:
            self.decision = random.choice(['buy', 'hold', 'sell'])

    def place_order(self, order_book, market_trend, current_round):
        if self.decision == 'buy':
            quantity = self.determine_buy_quantity()
            if quantity > 0:
                total_order_value = quantity * self.market_price
                if self.wealth >= total_order_value:  # Ensure enough wealth to place the order
                    order = {'type': 'buy', 'price': self.market_price, 'quantity': quantity, 'timestamp': current_round, 'agentid': self.id}
                    order_book.add_order(order)
                    self.wealth -= total_order_value  # Adjust wealth based on the order
                    self.reserved_wealth += total_order_value
        elif self.decision == 'sell':
            quantity = self.determine_sell_quantity()
            if quantity > 0 and self.coin_holdings >= quantity:
                order = {'type': 'sell', 'price': self.market_price, 'quantity': quantity, 'timestamp': current_round, 'agentid': self.id}
                order_book.add_order(order)
                self.coin_holdings -= quantity  # Temporarily adjust coin holdings
                self.reserved_coins += quantity 

    def determine_buy_quantity(self):
        # Allocate a percentage of wealth to purchase based on risk tolerance and market analysis
        spendable_wealth = self.wealth * self.risk_tolerance / self.market_price
        return min(spendable_wealth, self.wealth / self.market_price)  # Ensure not to spend more than current wealth

    def determine_sell_quantity(self):
        # Decide to sell based on risk tolerance and coin holdings
        if self.strategy == 'aggressive':
            # More aggressive strategy might sell a larger fraction of holdings
            return min(self.coin_holdings, self.coin_holdings * self.risk_tolerance)
        else:
            # More conservative or balanced approach
            return min(self.coin_holdings, self.coin_holdings * (self.risk_tolerance / 2))

    def update_performance_score(self, market_price):
        # Calculate performance score based on wealth change and market price
        new_wealth = self.wealth + (self.coin_holdings * market_price)
        self.performance_score = new_wealth - self.wealth  # Update based on wealth change

    def update_peers_based_on_performance(self, market):
        #Keep top performing peers, replace others
        if self.peers:
            # Sort peers by their performance score and keep the top half
            self.peers.sort(key=lambda peer: peer.performance_score, reverse=True)
            top_peers = self.peers[:len(self.peers)//2]
            
            # Fill the rest with new peers based on current performance rankings in the market
            market_peers = [agent for agent in market.agents if agent.id != self.id and agent not in self.peers]
            market_peers.sort(key=lambda agent: agent.performance_score, reverse=True)
            new_peers = market_peers[:len(self.peers) - len(top_peers)]
            
            self.peers = top_peers + new_peers
    
    def get_peer_influence(self):
        if not self.peers:
            return 0
        total_influence = 0
        total_weights = 0
        for peer in self.peers:
            # Calculate peer's influence weight based on their performance score, experience, and connectedness
            experience_weight = 1 + (peer.experience / 10)  # Normalize experience to a [1, 2] range
            connectedness_weight = len(peer.peers) / len(self.peers)  # Ratio of peer's connections to agent's connections
            performance_weight = peer.performance_score if peer.performance_score > 0 else 0
            influence_weight = experience_weight * connectedness_weight * performance_weight
            
            # Calculate the direction of the influence (-1 for sell, 1 for buy)
            decision_influence = 1 if peer.decision == 'buy' else -1 if peer.decision == 'sell' else 0
            
            # Aggregate weighted influence
            total_influence += decision_influence * influence_weight
            total_weights += influence_weight

        if total_weights == 0:
            return 0  # Avoid division by zero

        # Normalize the total influence by the sum of weights to get the average influence
        average_influence = total_influence / total_weights
        return average_influence


    def evaluate_news(self, news_impact):
        news_type = 'neutral'
        if news_impact < -0.3:
            news_type = 'negative'
        elif news_impact > 0.3:
            news_type = 'positive'
        reaction_scale = {
            'conservative': {'negative': 0.2, 'neutral': 0.4, 'positive': 0.6},
            'aggressive': {'negative': 0.1, 'neutral': 0.5, 'positive': 0.8},
            'balanced': {'negative': 0.3, 'neutral': 0.5, 'positive': 0.7},
            'adaptive': {'negative': 0.4, 'neutral': 0.5, 'positive': 0.6}
        }
        experience_factor = self.experience / 10
        return reaction_scale[self.strategy][news_type] * experience_factor


class Market:
    def __init__(self, num_agents, total_supply=1900, initial_market_price=13721.88, min_wealth=1000, alpha=1.16):
        self.total_supply = total_supply
        self.market_price = initial_market_price
        self.market_trend = random.choice(['bullish', 'bearish'])
        self.remaining_supply = total_supply
        self.volatility = 0.05
        self.order_book = OrderBook()
        self.agents = []
        self.price_history = [initial_market_price]
        self.round_summary = []
        self.trade_volume_history = []  # To track trade volume each round
        self.order_book_depth_history = []  # To track order book depth each round
        self.news_impact_history = []  # To track news impact each round
        self.current_round = 0
        self.wealth_snapshots = {}
        self.mining_rewards_log = []  # Store mining reward logs

        # Initial wealth and coin distribution
        wealth_distribution = (np.random.pareto(alpha, num_agents) + 1) * min_wealth
        coin_distribution = self.initial_coin_distribution(num_agents, total_supply)

        self.initial_mining_reward = 0.05 * self.remaining_supply

        # Initialize agents with diverse strategies and attributes
        for i in range(num_agents):
            agent = Agent(i, random.uniform(0.3, 0.7), random.randint(1, 10), 
                          random.choice(['conservative', 'aggressive', 'balanced', 'adaptive']),
                          wealth_distribution[i], coin_distribution[i], initial_market_price)
            self.agents.append(agent)

        self.setup_peer_network()

    def calculate_herding_index(self):
        buy_decisions = sum(1 for agent in self.agents if agent.decision == 'buy')
        sell_decisions = sum(1 for agent in self.agents if agent.decision == 'sell')
        total_decisions = len(self.agents)
        herding_direction = max(buy_decisions, sell_decisions)
        herding_index = herding_direction / total_decisions
        return herding_index

    def initial_coin_distribution(self, num_agents, total_supply):
        skewed_distribution = np.random.pareto(a=1.16, size=num_agents)
        scaled_distribution = skewed_distribution / skewed_distribution.sum() * total_supply
        return np.round(scaled_distribution).astype(int)

    def setup_peer_network(self, seed=1234):
        # Link agents based on similar strategies initially
        self.peer_network = nx.barabasi_albert_graph(num_agents, 3, seed=seed)  # Assuming each new node attaches to 3 existing nodes
        
        # Assign peers based on the network structure
        for agent in self.agents:
            agent.peers = [self.agents[peer_id] for peer_id in self.peer_network.neighbors(agent.id)]

    def update_peer_networks_based_on_performance(self):
        # Each agent updates its peer network based on performance
        for agent in self.agents:
            agent.update_peers_based_on_performance(self)

    
    def generate_news(self):
        return random.uniform(-1, 1)
    
    def distribute_mining_reward(self):
        mining_reward = self.initial_mining_reward / 365  # Distribute evenly across rounds in a year
        if self.remaining_supply >= mining_reward:
            lucky_trader = random.choice(self.agents)
            lucky_trader.coin_holdings += mining_reward
            self.remaining_supply -= mining_reward
            # Log the reward distribution
            self.mining_rewards_log.append({"Round": self.current_round, "Agent ID": lucky_trader.id, "Mining Reward": mining_reward})
        else:
            # Log when no rewards are available
            self.mining_rewards_log.append({"Round": self.current_round, "Agent ID": None, "Mining Reward": 0})

    def save_mining_rewards_log_to_excel(self, filepath='mining_rewards_log.xlsx'):
        if self.mining_rewards_log:  # Check if there are any logs to save
            df_mining_rewards_log = pd.DataFrame(self.mining_rewards_log)
            df_mining_rewards_log.to_excel(filepath, index=False)
        else:
            print("No mining rewards log to save.")

    
    
    def capture_wealth_snapshot(self, round_num):
        """Captures the current wealth of all agents at the given round."""
        self.wealth_snapshots[round_num] = {agent.id: agent.wealth for agent in self.agents}

    def save_wealth_snapshots_to_excel(self, filepath='wealth_snapshots.xlsx'):
        """Saves the wealth snapshots to an Excel file."""
        # Transforming the wealth snapshots into a format suitable for DataFrame construction
        data = {
            'Round': [],
            'Agent ID': [],
            'Wealth': []
        }
        for round_num, snapshot in self.wealth_snapshots.items():
            for agent_id, wealth in snapshot.items():
                data['Round'].append(round_num)
                data['Agent ID'].append(agent_id)
                data['Wealth'].append(wealth)
        
        df = pd.DataFrame(data)
        df.to_excel(filepath, index=False)

    def simulate(self, rounds):
        trend_history, volatility_history, price_history, herding_index_history, peer_influence_history = [], [], [], [], []
        decision_matrix = np.zeros((len(self.agents), rounds))

        for round_num in tqdm(range(rounds)):
            self.current_round = round_num
            if round_num % 50 == 0:  # Update peer networks every 50 rounds
                self.update_peer_networks_based_on_performance()
                self.distribute_mining_reward()
            self.market_news = self.generate_news()

            buy_count, sell_count = 0, 0
            for agent in self.agents:
                agent.make_decision(self.market_price, self.market_trend, self.market_news, self.volatility)
                agent.place_order(self.order_book, self.market_trend, round_num)
                valid_decision = agent.decision if agent.decision in ['buy', 'sell', 'hold'] else 'hold'
                decision_matrix[agent.id, round_num] = {'buy': 1, 'sell': -1, 'hold': 0}[valid_decision]
            

                if valid_decision == 'buy':
                    buy_count += 1
                elif valid_decision == 'sell':
                    sell_count += 1

                agent.update_performance_score(self.market_price)

            
            

            # Execute trades
            herding_index = self.calculate_herding_index()

            # Modify the match_orders call to pass the herding index
            trades = self.order_book.match_orders(herding_index)
            self.update_agent_states(trades)

            # Market update logic
            self.update_market_conditions(buy_count, sell_count)

            # Track trade volume for the round
            round_trade_volume = sum(trade['quantity'] for trade in self.order_book.trade_history)
            self.trade_volume_history.append(round_trade_volume)
            
            # Track order book depth
            buy_orders_depth = len(self.order_book.buy_orders)
            sell_orders_depth = len(self.order_book.sell_orders)
            self.order_book_depth_history.append((buy_orders_depth, sell_orders_depth))
            
            # Record news impact
            self.news_impact_history.append(self.market_news)

            # Calculate CSAD from the decision matrix
            average_decision = np.mean(decision_matrix[:, round_num])
            absolute_deviations = np.abs(decision_matrix[:, round_num] - average_decision)
            csad = np.mean(absolute_deviations)
            #Calculate CSSD from the decision matrix
            squared_deviations = (decision_matrix[:, round_num] - average_decision) ** 2
            cssd = np.mean(squared_deviations)
            # Record keeping for analysis
            price_history.append(self.market_price)
            trend_history.append(self.market_trend)
            volatility_history.append(self.volatility)
            herding_index_history.append(max(buy_count, sell_count) / len(self.agents))
            peer_influence_history.append(np.mean([a.get_peer_influence() for a in self.agents]))
            self.round_summary.append({
                'Round': round_num + 1,
                'Market Price': self.market_price,
                'Market Trend': self.market_trend,
                'Volatility': self.volatility,
                'Herding Index': max(buy_count, sell_count) / len(self.agents),
                'Average Peer Influence': np.mean([a.get_peer_influence() for a in self.agents]),
                'Trade Volume': round_trade_volume,
                'Buy Liquidity': self.order_book.total_buy_liquidity,
                'Sell Liquidity': self.order_book.total_sell_liquidity,
                'Buy Orders Quantity': buy_orders_depth,
                'Sell Orders Quantity': sell_orders_depth,
                'Buy Count': buy_count,
                'Sell Count': sell_count,
                'News Impact': self.market_news,
                'CSAD': csad,
                'CSSD': cssd
            })

        return trend_history, volatility_history, price_history, herding_index_history, peer_influence_history, decision_matrix

    def update_market_conditions(self, buy_count, sell_count):
        delta = (buy_count - sell_count) / len(self.agents)
        self.market_price *= (1 + delta * self.volatility)
        self.market_trend = 'bullish' if delta > 0 else 'bearish' if delta < 0 else self.market_trend
        self.price_history.append(self.market_price)
        self.volatility = self.calculate_volatility()

    def calculate_volatility(self):
    # Historical volatility: Use standard deviation of percentage price changes over the last N prices
        window_size = min(len(self.price_history), 10)  # Use last 10 prices, or fewer if not available
        if window_size <= 1:
            return self.volatility  # Return current volatility if not enough history
        
        # Adjust list comprehension to safely handle list length
        price_changes = [(self.price_history[i] / self.price_history[i-1] - 1) for i in range(1, len(self.price_history))][-window_size:]
        historical_volatility = np.std(price_changes)
        
        # News impact adjustment: Assume self.market_news impact is normalized between -1 and 1
        news_impact = abs(self.market_news)  # Absolute value to reflect impact magnitude, not direction
        
        # Combine historical volatility and news impact
        # Adjust weighting factors as necessary to reflect their relative importance
        combined_volatility = (0.7 * historical_volatility) + (0.3 * news_impact)
        
        # Ensure volatility stays within realistic bounds
        return min(max(combined_volatility, 0.1), 1.0)
    def update_agent_states(self, trades):
        if trades is None:
            trades = []  # Ensure trades is a list if None was returned
            print("No trades executed in the last round.")
        for trade in trades:
            buy_agent_id, sell_agent_id = trade['buyer_id'], trade['seller_id']
            buy_agent = next((agent for agent in self.agents if agent.id == buy_agent_id), None)
            sell_agent = next((agent for agent in self.agents if agent.id == sell_agent_id), None)
            
            if buy_agent and sell_agent:
                # Update the states of buying and selling agents based on the trade details
                buy_agent.coin_holdings += trade['quantity']
                buy_agent.reserved_wealth -= trade['quantity'] * trade['price']  # Adjusting reserved wealth
                
                sell_agent.reserved_coins -= trade['quantity']  # Adjusting reserved coins
                sell_agent.wealth += trade['quantity'] * trade['price']

    def save_market_summary_to_excel(self, filepath='market_summary.xlsx'):
        """Saves the market summary to an Excel file."""
        if self.round_summary:  # Check if there is any summary data to save
            df_market_summary = pd.DataFrame(self.round_summary)
            df_market_summary.to_excel(filepath, index=False)
        else:
            print("No market summary to save.")



# Simulation parameters
num_agents = 1000
num_rounds = 365

# Create and run the market simulation
market = Market(num_agents)
trend_history, volatility_history, price_history, herding_index_history, peer_influence_history, decision_matrix = market.simulate(num_rounds)

market.order_book.save_trade_history_to_excel('trade_history.xlsx')
market.order_book.save_orders_to_excel('orders.xlsx')
market.save_market_summary_to_excel('market_summary.xlsx')
market.save_mining_rewards_log_to_excel('mining_rewards_log.xlsx')


# Plotting the results
fig, axes = plt.subplots(4, 1, figsize=(12, 16))

# Market Trend Plot
axes[0].plot(trend_history, label='Market Trend')
axes[0].set_xlabel('Round')
axes[0].set_ylabel('Market Trend')
axes[0].set_title('Market Trend Over Time')
axes[0].legend()

# Market Volatility Plot
axes[1].plot(volatility_history, label='Market Volatility', color='red')
axes[1].set_xlabel('Round')
axes[1].set_ylabel('Volatility')
axes[1].set_title('Market Volatility Over Time')
axes[1].legend()

# Herding Index Plot
axes[2].plot(herding_index_history, label='Herding Index', color='green')
axes[2].set_xlabel('Round')
axes[2].set_ylabel('Herding Index')
axes[2].set_title('Herding Index Over Time')
axes[2].legend()

# Peer Influence vs. Decision Plot
axes[3].plot(peer_influence_history, label='Average Peer Influence', color='purple')
axes[3].set_xlabel('Round')
axes[3].set_ylabel('Average Peer Influence')
axes[3].set_title('Average Peer Influence Over Time')
axes[3].legend()

plt.tight_layout()
plt.show()

# Heatmap of Agent Decisions
plt.figure(figsize=(15, 10))
sns.heatmap(decision_matrix, cmap='coolwarm', cbar_kws={'label': 'Decision (Buy=1, Hold=0, Sell=-1)'})
plt.title('Heatmap of Agent Decisions')
plt.xlabel('Round')
plt.ylabel('Agent ID')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(price_history, label='Asset Price')
plt.xlabel('Round')
plt.ylabel('Price')
plt.title('Asset Price Trend Over Time')
plt.legend()
plt.show()

# At the end of the simulation, after all rounds have been executed:
final_market_price = market.market_price

for agent in market.agents:
    agent.final_wealth = agent.calculate_final_wealth(final_market_price)
    print(f"Agent {agent.id} final wealth: {agent.final_wealth}")

# Calculate and log the final wealth for each agent for verification
for agent in market.agents:
    # Calculate final wealth including cash, value of coin holdings, reserved wealth, and value of reserved coins
    agent.final_total_wealth = agent.wealth + (agent.coin_holdings * final_market_price) + agent.reserved_wealth + (agent.reserved_coins * final_market_price)

data = {
    "Agent ID": [agent.id for agent in market.agents],
    "Initial Wealth": [agent.initial_wealth for agent in market.agents],
    "Final Cash Wealth": [agent.wealth for agent in market.agents],
    "Initial Wealth with Coins": [agent.total_wealth for agent in market.agents],
    "Final Wealth with Coins": [agent.final_wealth for agent in market.agents],
    "Reserved Wealth": [agent.reserved_wealth for agent in market.agents],
    "Reserved Coins": [agent.reserved_coins for agent in market.agents],
    "Total Final Wealth": [agent.final_total_wealth for agent in market.agents],  # Add this line
    "Strategy": [agent.strategy for agent in market.agents],
    "Experience": [agent.experience for agent in market.agents],
    "Performance Score": [agent.performance_score for agent in market.agents],
    "Number of Peers": [len(agent.peers) for agent in market.agents],
    "Profitable": [agent.final_total_wealth > agent.total_wealth for agent in market.agents]

}

df = pd.DataFrame(data)

# Save to Excel
excel_path = 'agents_wealth.xlsx'
df.to_excel(excel_path, index=False)

final_wealths = [agent.final_wealth for agent in market.agents]

# Assuming 'market' is your Market instance and 'peer_network' is the network you've constructed
peer_network = market.peer_network  # Access the peer network

market.save_wealth_snapshots_to_excel('wealth_snapshots.xlsx')


# Define colors for each strategy
strategy_colors = {
    "conservative": "blue",
    "aggressive": "red",
    "balanced": "green",
    "adaptive": "purple"
}

# Map each agent's strategy to a color
node_colors = [strategy_colors[agent.strategy] for agent in market.agents]

# Apply the spring layout with a larger distance between nodes
pos = nx.spring_layout(peer_network, k=0.15, iterations=20)

plt.figure(figsize=(18, 12))  # Increase figure size to spread nodes out
nx.draw(peer_network, pos, with_labels=True, node_color=node_colors, node_size=50, font_size=8)
plt.title('Peer Network Visualization')
plt.show()