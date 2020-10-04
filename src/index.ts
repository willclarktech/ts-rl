import { Agent, DQN, Random, Reinforce } from "./agents";
import { Blackjack, CartPole, Environment } from "./environments";
import * as options from "./options";
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

		const didWin =
			env.winningScore !== undefined &&
			episode >= rollingAveragePeriod &&
			rollingAverageReturn >= env.winningScore;

		if (episode % logPeriod === 0 || didWin) {
			logEpisode(
				episode,
				returns,
				rollingAveragePeriod,
				rollingAverageReturns,
				logFile,
			);
		}

		if (didWin) {
			return true;
		}
	}

	return false;
};

const createEnv = (environmentName: string): Environment => {
	switch (environmentName) {
		case "blackjack":
			return new Blackjack();
		case "cartpole":
			return new CartPole();
		default:
			throw new Error("Environment name not recognised");
	}
};

const verifyAgentOptions = (
	agentOptions: unknown,
	agentName: string,
	environmentName: string,
): void => {
	if (agentOptions === undefined) {
		throw new Error(
			`Options not specified for ${agentName} in ${environmentName}`,
		);
	}
};

const createAgent = (agentName: string, env: Environment): Agent => {
	switch (agentName) {
		case "dqn": {
			const dqnOptions = options.DQN[env.name];
			verifyAgentOptions(dqnOptions, agentName, env.name);
			return new DQN(env, dqnOptions);
		}
		case "random": {
			return new Random(env);
		}
		case "reinforce": {
			const reinforceOptions = options.Reinforce[env.name];
			verifyAgentOptions(reinforceOptions, agentName, env.name);
			return new Reinforce(env, reinforceOptions);
		}
		default:
			throw new Error("Agent name not recognised");
	}
};

const main = (): void => {
	const agentName = process.argv[2] ?? "reinforce";
	const environmentName = process.argv[3] ?? "cartpole";

	const env = createEnv(environmentName);
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

main();
