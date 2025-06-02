import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import random
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class BlackjackEnvironment:
    """
    Comprehensive blackjack environment that simulates casino conditions
    """
    
    def __init__(self, num_decks=6, shuffle_threshold=0.25):
        self.num_decks = num_decks
        self.shuffle_threshold = shuffle_threshold
        self.reset_deck()
        
    def reset_deck(self):
        """Reset and shuffle the deck"""
        # Create deck: 1-10, J=10, Q=10, K=10 for each suit
        deck = []
        for _ in range(self.num_decks):
            for _ in range(4):  # 4 suits
                deck.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
        
        np.random.shuffle(deck)
        self.deck = deque(deck)
        self.cards_dealt = 0
        
    def deal_card(self):
        """Deal a single card from the deck"""
        if len(self.deck) / (self.num_decks * 52) < self.shuffle_threshold:
            self.reset_deck()
        
        self.cards_dealt += 1
        return self.deck.popleft()
    
    def hand_value(self, hand):
        """Calculate the value of a hand, handling aces optimally"""
        value = sum(hand)
        aces = hand.count(1)
        
        # Use aces as 11 when beneficial
        while aces > 0 and value + 10 <= 21:
            value += 10
            aces -= 1
            
        return value
    
    def has_usable_ace(self, hand):
        """Check if hand has an ace being used as 11"""
        return 1 in hand and self.hand_value(hand) != sum(hand)
    
    def is_blackjack(self, hand):
        """Check if hand is a blackjack (21 with 2 cards)"""
        return len(hand) == 2 and self.hand_value(hand) == 21
    
    def get_state(self, player_hand, dealer_upcard, can_double=True, can_split=False):
        """Get the current state representation"""
        player_sum = self.hand_value(player_hand)
        usable_ace = self.has_usable_ace(player_hand)
        return (player_sum, dealer_upcard, usable_ace, can_double, can_split)
    
    def get_valid_actions(self, player_hand, can_double=True, can_split=False):
        """Get list of valid actions for current state"""
        actions = ['hit', 'stand']
        
        if can_double and len(player_hand) == 2:
            actions.append('double')
            
        if can_split and len(player_hand) == 2 and player_hand[0] == player_hand[1]:
            actions.append('split')
            
        return actions
    
    def play_dealer(self, dealer_hand):
        """Play out dealer's hand according to casino rules"""
        while self.hand_value(dealer_hand) < 17:
            dealer_hand.append(self.deal_card())
        return dealer_hand
    
    def calculate_reward(self, player_hands, dealer_hand, doubled=False):
        """Calculate reward for the hand"""
        dealer_value = self.hand_value(dealer_hand)
        dealer_busted = dealer_value > 21
        
        total_reward = 0
        
        for hand in player_hands:
            player_value = self.hand_value(hand)
            bet_multiplier = 2 if doubled else 1
            
            if player_value > 21:
                # Player busted
                reward = -1 * bet_multiplier
            elif dealer_busted:
                # Dealer busted, player wins
                if self.is_blackjack(hand):
                    reward = 1.5 * bet_multiplier  # Blackjack pays 3:2
                else:
                    reward = 1 * bet_multiplier
            elif player_value > dealer_value:
                # Player wins
                if self.is_blackjack(hand):
                    reward = 1.5 * bet_multiplier
                else:
                    reward = 1 * bet_multiplier
            elif player_value < dealer_value:
                # Dealer wins
                reward = -1 * bet_multiplier
            else:
                # Push
                reward = 0
                
            total_reward += reward
            
        return total_reward

class RLAgent:
    """Base class for reinforcement learning agents"""
    
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
    def choose_action(self, state, valid_actions):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            return self.get_best_action(state, valid_actions)
    
    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_best_action(self, state, valid_actions):
        """Get best action according to current policy (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def update(self, *args):
        """Update the agent (to be implemented by subclasses)"""
        raise NotImplementedError

class MonteCarloAgent(RLAgent):
    """Monte Carlo reinforcement learning agent"""
    
    def __init__(self, first_visit=True, **kwargs):
        super().__init__(**kwargs)
        self.Q = defaultdict(lambda: defaultdict(float))
        self.returns = defaultdict(lambda: defaultdict(list))
        self.first_visit = first_visit
        
    def get_best_action(self, state, valid_actions):
        """Get action with highest Q-value"""
        q_values = [self.Q[state][action] for action in valid_actions]
        if all(q == 0 for q in q_values):
            return np.random.choice(valid_actions)
        
        best_idx = np.argmax(q_values)
        return valid_actions[best_idx]
    
    def update_from_episode(self, episode):
        """Update Q-values from a complete episode"""
        states, actions, rewards = episode
        G = 0
        
        # Work backwards through the episode
        for t in reversed(range(len(states))):
            G = rewards[t] + self.gamma * G
            state_action = (states[t], actions[t])
            
            # First-visit Monte Carlo
            if self.first_visit:
                if state_action not in [(states[i], actions[i]) for i in range(t)]:
                    self.returns[states[t]][actions[t]].append(G)
                    self.Q[states[t]][actions[t]] = np.mean(self.returns[states[t]][actions[t]])
            else:
                # Every-visit Monte Carlo
                self.returns[states[t]][actions[t]].append(G)
                self.Q[states[t]][actions[t]] = np.mean(self.returns[states[t]][actions[t]])

class QLearningAgent(RLAgent):
    """Q-Learning agent with tabular representation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Q = defaultdict(lambda: defaultdict(float))
        
    def get_best_action(self, state, valid_actions):
        """Get action with highest Q-value"""
        q_values = [self.Q[state][action] for action in valid_actions]
        if all(q == 0 for q in q_values):
            return np.random.choice(valid_actions)
        
        best_idx = np.argmax(q_values)
        return valid_actions[best_idx]
    
    def update(self, state, action, reward, next_state, next_valid_actions, done):
        """Update Q-value using Q-learning rule"""
        current_q = self.Q[state][action]
        
        if done:
            max_next_q = 0
        else:
            next_q_values = [self.Q[next_state][a] for a in next_valid_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
        
        # Q-learning update rule
        self.Q[state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

class NeuralQLearningAgent(RLAgent):
    """Q-Learning agent with neural network function approximation"""
    
    def __init__(self, state_size=5, action_size=4, hidden_sizes=[128, 64], **kwargs):
        super().__init__(**kwargs)
        self.state_size = state_size
        self.action_size = action_size
        self.action_map = {'hit': 0, 'stand': 1, 'double': 2, 'split': 3}
        self.reverse_action_map = {v: k for k, v in self.action_map.items()}
        
        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = self._build_network(hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Experience replay
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        
    def _build_network(self, hidden_sizes):
        """Build the neural network"""
        layers = []
        prev_size = self.state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, self.action_size))
        
        return nn.Sequential(*layers)
    
    def state_to_tensor(self, state):
        """Convert state to tensor"""
        player_sum, dealer_card, usable_ace, can_double, can_split = state
        features = [
            player_sum / 21.0,  # Normalize player sum
            dealer_card / 10.0,  # Normalize dealer card
            float(usable_ace),
            float(can_double),
            float(can_split)
        ]
        return torch.FloatTensor(features).to(self.device)
    
    def get_best_action(self, state, valid_actions):
        """Get best action using neural network"""
        state_tensor = self.state_to_tensor(state).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        # Filter to only valid actions
        valid_indices = [self.action_map[action] for action in valid_actions if action in self.action_map]
        
        if not valid_indices:
            return np.random.choice(valid_actions)
        
        valid_q_values = [q_values[i] for i in valid_indices]
        best_idx = valid_indices[np.argmax(valid_q_values)]
        
        return self.reverse_action_map[best_idx]
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        # Filter out experiences with None next_states and invalid actions
        valid_batch = []
        for experience in batch:
            state, action, reward, next_state, done = experience
            if action in self.action_map and (next_state is not None or done):
                valid_batch.append(experience)
        
        if len(valid_batch) < self.batch_size // 2:
            return
        
        states = torch.stack([self.state_to_tensor(experience[0]) for experience in valid_batch])
        actions = torch.LongTensor([self.action_map[experience[1]] for experience in valid_batch]).to(self.device)
        rewards = torch.FloatTensor([experience[2] for experience in valid_batch]).to(self.device)
        dones = torch.BoolTensor([experience[4] for experience in valid_batch]).to(self.device)
        
        # Handle next states carefully
        next_q_values = torch.zeros(len(valid_batch)).to(self.device)
        for i, experience in enumerate(valid_batch):
            if not experience[4] and experience[3] is not None:  # Not done and has next state
                next_state_tensor = self.state_to_tensor(experience[3]).unsqueeze(0)
                next_q_values[i] = self.q_network(next_state_tensor).max(1)[0].detach()
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class BlackjackTrainer:
    """Training and evaluation framework for blackjack agents"""
    
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.training_history = []
        
    def play_episode(self, training=True):
        """Play a single episode"""
        # Deal initial cards
        player_hand = [self.env.deal_card(), self.env.deal_card()]
        dealer_hand = [self.env.deal_card(), self.env.deal_card()]
        dealer_upcard = dealer_hand[0]
        
        # Check for immediate blackjacks
        player_blackjack = self.env.is_blackjack(player_hand)
        dealer_blackjack = self.env.is_blackjack(dealer_hand)
        
        if player_blackjack and dealer_blackjack:
            return 0  # Push
        elif player_blackjack:
            return 1.5  # Blackjack wins 3:2
        elif dealer_blackjack:
            return -1  # Dealer blackjack
        
        # Play the hand
        episode_data = []
        done = False
        doubled = False
        busted = False
        
        while not done:
            state = self.env.get_state(player_hand, dealer_upcard, 
                                     can_double=len(player_hand)==2, 
                                     can_split=False)  # Simplified: no splitting for now
            valid_actions = self.env.get_valid_actions(player_hand, 
                                                     can_double=len(player_hand)==2,
                                                     can_split=False)
            
            action = self.agent.choose_action(state, valid_actions)
            
            if action == 'hit':
                player_hand.append(self.env.deal_card())
                
                # Check if player busted
                if self.env.hand_value(player_hand) > 21:
                    done = True
                    busted = True
                    
            elif action == 'stand':
                done = True
                
            elif action == 'double':
                doubled = True
                player_hand.append(self.env.deal_card())
                done = True
                # Check if player busted after doubling
                if self.env.hand_value(player_hand) > 21:
                    busted = True
            
            # Store experience for learning
            if training:
                episode_data.append((state, action))
        
        # Calculate final reward
        if busted:
            # Player busted - automatic loss
            final_reward = -2 if doubled else -1
        else:
            # Player didn't bust - play dealer and compare
            dealer_hand = self.env.play_dealer(dealer_hand)
            
            player_value = self.env.hand_value(player_hand)
            dealer_value = self.env.hand_value(dealer_hand)
            bet_multiplier = 2 if doubled else 1
            
            if dealer_value > 21:
                # Dealer busted, player wins
                final_reward = 1 * bet_multiplier
            elif player_value > dealer_value:
                # Player wins
                final_reward = 1 * bet_multiplier
            elif player_value < dealer_value:
                # Dealer wins
                final_reward = -1 * bet_multiplier
            else:
                # Push
                final_reward = 0
        
        # Update agent based on episode
        if training and episode_data:
            if isinstance(self.agent, MonteCarloAgent):
                # For Monte Carlo, propagate final reward to all steps
                states, actions = zip(*episode_data)
                rewards = [final_reward] * len(states)  # Same final reward for all steps
                self.agent.update_from_episode((states, actions, rewards))
            
            elif isinstance(self.agent, (QLearningAgent, NeuralQLearningAgent)):
                # For Q-learning, update each step
                for i, (state, action) in enumerate(episode_data):
                    is_final = (i == len(episode_data) - 1)
                    next_state = None if is_final else episode_data[i+1][0]
                    step_reward = final_reward if is_final else 0
                    
                    if isinstance(self.agent, NeuralQLearningAgent):
                        self.agent.remember(state, action, step_reward, next_state, is_final)
                    else:
                        next_valid_actions = ['hit', 'stand'] if not is_final else []
                        self.agent.update(state, action, step_reward, next_state, next_valid_actions, is_final)
            
            if hasattr(self.agent, 'update_epsilon'):
                self.agent.update_epsilon()
        
        return final_reward
    
    def train(self, episodes=1000000, eval_interval=50000, eval_episodes=10000):
        """Train the agent"""
        print(f"Training {type(self.agent).__name__} for {episodes} episodes...")
        
        for episode in range(episodes):
            reward = self.play_episode(training=True)
            
            # Neural network replay
            if isinstance(self.agent, NeuralQLearningAgent) and episode % 4 == 0:
                self.agent.replay()
            
            # Evaluation
            if episode % eval_interval == 0:
                eval_reward = self.evaluate(eval_episodes)
                self.training_history.append((episode, eval_reward))
                print(f"Episode {episode}: Avg reward = {eval_reward:.4f}, Epsilon = {self.agent.epsilon:.4f}")
    
    def evaluate(self, episodes=100000):
        """Evaluate the agent's performance"""
        # Handle agents that don't have epsilon (like BasicStrategyAgent)
        original_epsilon = getattr(self.agent, 'epsilon', 0)
        if hasattr(self.agent, 'epsilon'):
            self.agent.epsilon = 0  # No exploration during evaluation
        
        rewards = []
        wins = losses = pushes = 0
        
        for _ in range(episodes):
            reward = self.play_episode(training=False)
            rewards.append(reward)
            
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                pushes += 1
        
        if hasattr(self.agent, 'epsilon'):
            self.agent.epsilon = original_epsilon
        
        avg_reward = np.mean(rewards)
        print(f"   Detailed stats: W:{wins/episodes*100:.1f}% L:{losses/episodes*100:.1f}% P:{pushes/episodes*100:.1f}%")
        
        return avg_reward

class BasicStrategyAgent:
    """Perfect basic strategy agent for comparison"""
    
    def __init__(self):
        self.epsilon = 0  # Add epsilon attribute for compatibility
        # Simplified basic strategy (hard totals only)
        self.strategy = {
            # (player_sum, dealer_upcard): action
            **{(i, j): 'hit' for i in range(5, 12) for j in range(1, 11)},
            **{(12, j): 'hit' if j in [2, 3, 7, 8, 9, 10, 1] else 'stand' for j in range(1, 11)},
            **{(i, j): 'hit' if j in [7, 8, 9, 10, 1] else 'stand' for i in [13, 14, 15, 16] for j in range(1, 11)},
            **{(i, j): 'stand' for i in range(17, 22) for j in range(1, 11)}
        }
    
    def choose_action(self, state, valid_actions):
        """Choose action based on basic strategy"""
        player_sum, dealer_upcard, usable_ace, can_double, can_split = state
        
        # Simplified: ignore soft hands and doubling for now
        key = (player_sum, dealer_upcard)
        
        if key in self.strategy:
            action = self.strategy[key]
            if action in valid_actions:
                return action
        
        # Default to stand if strategy not found or action not valid
        return 'stand' if 'stand' in valid_actions else 'hit'

class BlackjackVisualizer:
    """Visualization tools for blackjack AI analysis"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
    
    def plot_training_curves(self, histories, labels):
        """Plot training curves for multiple agents"""
        plt.figure(figsize=(12, 6))
        
        for history, label in zip(histories, labels):
            if history:
                episodes, rewards = zip(*history)
                plt.plot(episodes, rewards, label=label, linewidth=2)
        
        plt.xlabel('Training Episodes')
        plt.ylabel('Average Reward')
        plt.title('Training Progress Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_strategy_heatmap(self, agent, title="Learned Strategy"):
        """Plot strategy heatmap"""
        strategy_matrix = np.zeros((18, 10))  # 18 player sums (4-21), 10 dealer cards
        
        for player_sum in range(4, 22):
            for dealer_card in range(1, 11):
                state = (player_sum, dealer_card, False, False, False)
                valid_actions = ['hit', 'stand']
                
                if hasattr(agent, 'get_best_action'):
                    action = agent.get_best_action(state, valid_actions)
                else:
                    action = agent.choose_action(state, valid_actions)
                
                # 0 = hit, 1 = stand
                strategy_matrix[player_sum - 4, dealer_card - 1] = 1 if action == 'stand' else 0
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(strategy_matrix, 
                   xticklabels=['A', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                   yticklabels=[str(i) for i in range(4, 22)],
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Action (0=Hit, 1=Stand)'},
                   annot=False)
        
        plt.title(f'{title} - Hard Totals')
        plt.xlabel('Dealer Upcard')
        plt.ylabel('Player Sum')
        plt.tight_layout()
        plt.show()
    
    def compare_performance(self, results, labels):
        """Compare final performance of different agents"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Expected returns
        returns = [result['expected_return'] for result in results]
        colors = plt.cm.viridis(np.linspace(0, 1, len(returns)))
        
        bars1 = ax1.bar(labels, returns, color=colors)
        ax1.set_ylabel('Expected Return per Hand')
        ax1.set_title('Expected Return Comparison')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break Even')
        ax1.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars1, returns):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Win rates
        win_rates = [result['win_rate'] for result in results]
        bars2 = ax2.bar(labels, win_rates, color=colors)
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('Win Rate Comparison')
        
        # Add value labels on bars
        for bar, value in zip(bars2, win_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the blackjack AI project"""
    print(" Blackjack Reinforcement Learning AI Project")
    print("=" * 50)
    
    # Initialize environment
    env = BlackjackEnvironment(num_decks=6)
    
    # Quick test of environment to verify reward logic
    print(" Testing environment logic...")
    test_rewards = []
    for _ in range(1000):
        # Simulate a basic random episode
        player_hand = [env.deal_card(), env.deal_card()]
        dealer_hand = [env.deal_card(), env.deal_card()]
        
        # Simple random play
        while env.hand_value(player_hand) < 17:
            player_hand.append(env.deal_card())
        
        if env.hand_value(player_hand) <= 21:
            dealer_hand = env.play_dealer(dealer_hand)
            reward = env.calculate_reward([player_hand], dealer_hand, False)
        else:
            reward = -1  # Player busted
        
        test_rewards.append(reward)
    
    avg_test_reward = np.mean(test_rewards)
    print(f"Environment test - Average reward: {avg_test_reward:.4f}")
    print(f"Should be negative (house edge). If positive, there's a bug.")
    
    # Initialize agents
    agents = {
        'Monte Carlo (First-Visit)': MonteCarloAgent(first_visit=True, epsilon_decay=0.99995),
        'Q-Learning': QLearningAgent(epsilon_decay=0.99995),
        'Neural Q-Learning': NeuralQLearningAgent(epsilon_decay=0.99995)
    }
    
    # Basic strategy for comparison
    basic_strategy = BasicStrategyAgent()
    
    # Training parameters
    training_episodes = 100000  # Reduced for faster debugging
    eval_episodes = 10000
    
    # Train agents and collect results
    results = []
    histories = []
    labels = []
    
    for name, agent in agents.items():
        print(f"\n Training {name}...")
        trainer = BlackjackTrainer(env, agent)
        trainer.train(episodes=training_episodes, eval_interval=50000, eval_episodes=eval_episodes)
        
        # Final evaluation
        print(f" Evaluating {name}...")
        final_performance = trainer.evaluate(episodes=50000)
        
        # Calculate detailed statistics for summary
        evaluation_rewards = []
        wins = losses = pushes = 0
        
        # Additional evaluation for detailed stats
        for _ in range(10000):
            reward = trainer.play_episode(training=False)
            evaluation_rewards.append(reward)
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                pushes += 1
        
        result = {
            'expected_return': final_performance,
            'win_rate': wins / 100,
            'loss_rate': losses / 100,
            'push_rate': pushes / 100
        }
        
        results.append(result)
        histories.append(trainer.training_history)
        labels.append(name)
        
        print(f" {name} Results:")
        print(f"   Expected Return: {result['expected_return']:.4f}")
        print(f"   Win Rate: {result['win_rate']:.1f}%")
        print(f"   Loss Rate: {result['loss_rate']:.1f}%")
        print(f"   Push Rate: {result['push_rate']:.1f}%")
    
    # Evaluate basic strategy
    print(f"\n Evaluating Basic Strategy...")
    basic_trainer = BlackjackTrainer(env, basic_strategy)
    basic_performance = basic_trainer.evaluate(episodes=50000)
    
    wins = losses = pushes = 0
    for _ in range(10000):
        reward = basic_trainer.play_episode(training=False)
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            pushes += 1
    
    basic_result = {
        'expected_return': basic_performance,
        'win_rate': wins / 100,
        'loss_rate': losses / 100,
        'push_rate': pushes / 100
    }
    
    results.append(basic_result)
    labels.append('Basic Strategy')
    
    print(f" Basic Strategy Results:")
    print(f"   Expected Return: {basic_result['expected_return']:.4f}")
    print(f"   Win Rate: {basic_result['win_rate']:.1f}%")
    print(f"   Loss Rate: {basic_result['loss_rate']:.1f}%")
    print(f"   Push Rate: {basic_result['push_rate']:.1f}%")
    
    # Summary table
    print(f"\n Final Results Summary:")
    print("=" * 80)
    df = pd.DataFrame(results, index=labels)
    print(df.round(4))
    
    # Expected returns should be negative!
    if any(r['expected_return'] > 0 for r in results):
        print("\n  WARNING: Positive expected returns detected - there may be a bug in the reward calculation!")
    else:
        print("\n All expected returns are negative as expected for blackjack.")
    
    print(f"\n Project Complete! All agents have been trained and evaluated.")

if __name__ == "__main__":
    main()
