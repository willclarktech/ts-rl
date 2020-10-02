import { Agent, DQN, Random, Reinforce } from "./agents";
import { Blackjack, CartPole, Environment } from "./environments";
import { getTimeString, logEpisode, mean } from "./util";

const train = (
	env: Environment,
	agent: Agent,
	maxEpisodes: number,
	rollingAveragePeriod: number,
	logPeriod: number,
	logFile: string,
): boolean => {
	const returns = [];
	const rollingAverageReturns = [];

	for (let episode = 1; episode <= maxEpisodes; ++episode) {
		const ret = agent.runEpisode(env);
		returns.push(ret);

		const rollingAverageReturn = mean(returns.slice(-100));
		rollingAverageReturns.push(rollingAverageReturn);

		if (episode % logPeriod === 0) {
			logEpisode(
				episode,
				returns,
				rollingAveragePeriod,
				rollingAverageReturns,
				logFile,
			);
		}

		if (
			env.winningScore !== undefined &&
			episode >= rollingAveragePeriod &&
			rollingAverageReturn >= env.winningScore
		) {
			return true;
		}
	}

	return false;
};

const envs: { readonly [key: string]: () => Environment } = {
	blackjack: (): Environment => new Blackjack(),
	cartpole: (): Environment => new CartPole(),
};

const createAgent = (agentName: string, env: Environment): Agent => {
	switch (agentName) {
		case "dqn": {
			const hiddenWidths = [8];
			const alpha = 0.0001;
			const gamma = 0.99;
			const epsilonInitial = 1;
			const epsilonMinimum = 0.01;
			const epsilonReduction = 0.001;
			const replayMemoryCapacity = 512;
			const minibatchSize = 32;
			return new DQN(
				env,
				hiddenWidths,
				alpha,
				gamma,
				epsilonInitial,
				epsilonMinimum,
				epsilonReduction,
				replayMemoryCapacity,
				minibatchSize,
			);
		}
		case "random": {
			return new Random(env);
		}
		case "reinforce": {
			const hiddenWidths = [24];
			const alpha = 0.01;
			const gamma = 0.99;
			return new Reinforce(env, hiddenWidths, alpha, gamma);
		}
		default:
			throw new Error("Agent name not recognised");
	}
};

const main = (): void => {
	const agentName = process.argv[2] ?? "reinforce";
	const environmentName = process.argv[3] ?? "cartpole";

	const createEnv: () => Environment = envs[environmentName];
	if (createEnv === undefined) {
		throw new Error("Environment name not recognised");
	}
	const env = createEnv();
	const agent = createAgent(agentName, env);

	const maxEpisodes = 1000;
	const rollingAveragePeriod = 100;
	const logPeriod = 10;
	const experimentName = `${agent.name}-${env.name}`;
	const logFile = `./results/data/${experimentName}.json`;

	console.info(
		`${getTimeString()} - Score to beat: ${env.winningScore ?? "[not set]"}`,
	);

	const didWin = train(
		env,
		agent,
		maxEpisodes,
		rollingAveragePeriod,
		logPeriod,
		logFile,
	);

	console.info(didWin ? "You won!" : "You lost.");
};

// Run with eg `npm start [Blackjack]`
main();
