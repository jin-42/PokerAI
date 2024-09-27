import os
import argparse
import torch
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from rlcard.agents import DQNAgent, NFSPAgent

# Définition d'un agent humain personnalisé
class HumanAgent:
    """Agent humain qui interagit via la console."""
    def __init__(self, action_num):
        self.action_num = action_num

    def step(self, state):
        # Afficher l'état (optionnel, dépend de l'environnement)
        print("\nÉtat actuel:", state)

        # Afficher les actions possibles
        print("Actions possibles:", list(range(self.action_num)))

        # Demander à l'utilisateur d'entrer une action
        while True:
            try:
                action = int(input("Entrez votre action: "))
                if 0 <= action < self.action_num:
                    return action
                else:
                    print(f"Veuillez entrer un nombre entre 0 et {self.action_num - 1}.")
            except ValueError:
                print("Entrée invalide. Veuillez entrer un nombre entier.")

    def eval_step(self, state):
        return self.step(state)

def train(args):
    # Vérifier la disponibilité du GPU
    device = get_device()

    # Définir la graine pour la reproductibilité
    set_seed(args.seed)

    # Créer l'environnement avec la graine spécifiée
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
            'single_agent_mode': False,  # Mode multi-agents
        }
    )

    # Initialiser l'agent et utiliser des agents plus forts comme adversaires
    if args.algorithm == 'dqn':
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            mlp_layers=[128, 128],  # Augmentation de la taille du réseau
            device=device,
            lr=1e-4,  # Taux d'apprentissage ajusté
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=50000,
            update_target_every=1000,
            memory_size=200000,  # Taille de la mémoire augmentée
            batch_size=64,
            gamma=0.99,
        )
    elif args.algorithm == 'nfsp':
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape[0],
            hidden_layers_sizes=[128, 128],  # Augmentation de la taille du réseau
            q_mlp_layers=[128, 128],
            device=device,
            hidden_layers_sizes_policy=[128, 128],
            hidden_layers_sizes_q=[128, 128],
            anticipatory_param=0.1,
            lr=1e-4,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=50000,
            memory_size=200000,
            batch_size=64,
            gamma=0.99,
        )
    else:
        raise ValueError("Algorithme non supporté: {}".format(args.algorithm))

    # Utiliser des adversaires plus forts (par exemple, plusieurs RandomAgents ou des agents pré-entraînés)
    opponents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players - 1)]
    agents = [agent] + opponents
    env.set_agents(agents)

    # Initialiser le logger
    with Logger(args.log_dir) as logger:
        best_performance = -float('inf')  # Initialiser la meilleure performance
        best_model_path = os.path.join(args.log_dir, 'best.pt')

        for episode in range(1, args.num_episodes + 1):
            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Générer des données depuis l'environnement
            trajectories, payoffs = env.run(is_training=True)

            # Réorganiser les données en (état, action, récompense, prochain état, terminé)
            trajectories = reorganize(trajectories, payoffs)

            # Alimenter les transitions dans la mémoire de l'agent et entraîner l'agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Évaluer les performances périodiquement
            if episode % args.evaluate_every == 0:
                win_rates = tournament(env, args.num_eval_games)
                performance = win_rates[0]
                logger.log_performance(episode, performance)
                print(f"Épisode {episode}: Performance = {performance:.4f}")

                # Sauvegarder le meilleur modèle
                if performance > best_performance:
                    best_performance = performance
                    torch.save(agent, best_model_path)
                    print(f"Meilleur modèle sauvegardé à l'épisode {episode} avec performance {performance:.4f}")

        # Obtenir les chemins des fichiers de logs
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Tracer la courbe d'apprentissage
    plot_curve(csv_path, fig_path, args.algorithm)

    print(f"Entraînement terminé. Meilleur modèle sauvegardé à {best_model_path}")

def play(args):
    # Charger le meilleur modèle
    model_path = os.path.join(args.log_dir, 'best.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_path} n'existe pas. Veuillez entraîner le modèle d'abord.")

    agent = torch.load(model_path)
    device = get_device()
    agent.device = device  # S'assurer que l'agent utilise le bon device
    if hasattr(agent, 'eval_mode'):
        agent.eval_mode()  # Mettre l'agent en mode évaluation si applicable

    # Créer l'environnement
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
            'single_agent_mode': False,
        }
    )

    # Définir les agents (l'agent entraîné contre un agent humain)
    human_agent = HumanAgent(num_actions=env.num_actions)
    env.set_agents([agent, human_agent])

    # Jouer des parties
    for game in range(1, args.num_play_games + 1):
        trajectories, payoffs = env.run(is_training=False)
        env.render()  # Afficher le jeu

        # Afficher le résultat
        if payoffs[0] > payoffs[1]:
            print(f"Partie {game}: L'agent entraîné a gagné !")
        elif payoffs[0] < payoffs[1]:
            print(f"Partie {game}: Vous avez gagné !")
        else:
            print(f"Partie {game}: Égalité !")

def evaluate(args):
    # Charger le meilleur modèle
    model_path = os.path.join(args.log_dir, 'best.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_path} n'existe pas. Veuillez entraîner le modèle d'abord.")

    agent = torch.load(model_path)
    device = get_device()
    agent.device = device
    if hasattr(agent, 'eval_mode'):
        agent.eval_mode()

    # Créer l'environnement
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
            'single_agent_mode': False,
        }
    )

    # Définir les agents (l'agent entraîné contre des agents aléatoires)
    opponents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players - 1)]
    env.set_agents([agent] + opponents)

    # Évaluer les performances
    win_rates = tournament(env, args.num_eval_games)
    print(f"Performance de l'agent entraîné sur {args.num_eval_games} parties: {win_rates[0]:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP amélioré dans RLCard")
    subparsers = parser.add_subparsers(dest='mode', help='Mode de fonctionnement')

    # Sous-parser pour l'entraînement
    train_parser = subparsers.add_parser('train', help='Entraîner l\'agent')
    train_parser.add_argument(
        '--env',
        type=str,
        default='no-limit-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )
    train_parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=[
            'dqn',
            'nfsp',
        ],
    )
    train_parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    train_parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    train_parser.add_argument(
        '--num_episodes',
        type=int,
        default=1000000,
    )
    train_parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    train_parser.add_argument(
        '--evaluate_every',
        type=int,
        default=1000,  # Augmentation de la fréquence d'évaluation
    )
    train_parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/no_limit_holdem_dqn_result/',
    )

    # Sous-parser pour jouer contre l'agent
    play_parser = subparsers.add_parser('play', help='Jouer contre l\'agent entraîné')
    play_parser.add_argument(
        '--env',
        type=str,
        default='no-limit-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )
    play_parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/no_limit_holdem_dqn_result/',
    )
    play_parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    play_parser.add_argument(
        '--num_play_games',
        type=int,
        default=1,
    )

    # Sous-parser pour évaluer les performances
    eval_parser = subparsers.add_parser('evaluate', help='Évaluer les performances de l\'agent entraîné')
    eval_parser.add_argument(
        '--env',
        type=str,
        default='no-limit-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )
    eval_parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/no_limit_holdem_dqn_result/',
    )
    eval_parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    eval_parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )

    args = parser.parse_args()

    if args.mode == 'train':
        os.makedirs(args.log_dir, exist_ok=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        train(args)
    elif args.mode == 'play':
        play(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    else:
        parser.print_help()
