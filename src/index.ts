import "./random";

import { ActorCritic, Agent, DQN, Random, Reinforce } from "./agents";
import { Blackjack, CartPole, Environment, MountainCar } from "./environments";
import * as options from "./options";
import { log, logEpisode, mean } from "./util";

const train = (
	environment: Environment,
	agent: Agent,
	{
		maxEpisodes,
		rollingAveragePeriod,
		logPeriod,
		logDirectory,
		warmupEpisodes,
	}: options.TrainingOptions,
): boolean => {
	const logFile = `${logDirectory}/${agent.name}-${environment.name}.json`;

	const returns = [];
	const rollingAverageReturns = [];

	log(`Warming up for ${warmupEpisodes} episodes...`);
	for (let episode = 1; episode <= warmupEpisodes; ++episode) {
		agent.runEpisode(environment, true);
	}
	log("Finished warming up");

	for (let episode = 1; episode <= maxEpisodes; ++episode) {
		const ret = agent.runEpisode(environment);
		returns.push(ret);

		const rollingAverageReturn = mean(returns.slice(-100));
		rollingAverageReturns.push(rollingAverageReturn);

		const didWin =
			environment.winningScore !== undefined &&
			episode >= rollingAveragePeriod &&
			rollingAverageReturn >= environment.winningScore;

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

const createEnvironment = (environmentName: string): Environment => {
	switch (environmentName) {
		case "blackjack":
			return new Blackjack();
		case "cart-pole":
			return new CartPole();
		case "mountain-car":
			return new MountainCar();
		default:
			throw new Error("Environment name not recognised");
	}
};

const verifyOptions = (
	agentOptions: unknown,
	agentName: string,
	environmentName: string,
): void => {
	if (agentOptions === undefined) {
		throw new Error(
			`Options not specified for ${agentName} in ${environmentName}`,
		);
	}
	log(
		`Using agent ${agentName} with options: ${JSON.stringify(
			agentOptions,
			undefined,
			"\t",
		)}`,
	);
};

const createAgentAndGetOptions = (
	agentName: string,
	environment: Environment,
): {
	readonly agent: Agent;
	readonly trainingOptions: options.TrainingOptions;
} => {
	switch (agentName) {
		case "actor-critic": {
			const actorCriticOptions = options.ActorCritic[environment.name];
			verifyOptions(actorCriticOptions, agentName, environment.name);
			const { trainingOptions, ...agentOptions } = actorCriticOptions;
			return {
				agent: new ActorCritic(environment, agentOptions),
				trainingOptions,
			};
		}
		case "dqn": {
			const dqnOptions = options.DQN[environment.name];
			verifyOptions(dqnOptions, agentName, environment.name);
			const { trainingOptions, ...agentOptions } = dqnOptions;
			return {
				agent: new DQN(environment, agentOptions),
				trainingOptions,
			};
		}
		case "random": {
			const randomOptions = options.Random[environment.name];
			verifyOptions(randomOptions, agentName, environment.name);
			const { trainingOptions } = randomOptions;
			return {
				agent: new Random(environment),
				trainingOptions,
			};
		}
		case "reinforce": {
			const reinforceOptions = options.Reinforce[environment.name];
			verifyOptions(reinforceOptions, agentName, environment.name);
			const { trainingOptions, ...agentOptions } = reinforceOptions;
			return {
				agent: new Reinforce(environment, agentOptions),
				trainingOptions,
			};
		}
		default:
			throw new Error("Agent name not recognised");
	}
};

const main = async (): Promise<void> => {
	const agentName = process.argv[2] ?? "reinforce";
	const environmentName = process.argv[3] ?? "cartpole";

	const environment = createEnvironment(environmentName);
	const { agent, trainingOptions } = createAgentAndGetOptions(
		agentName,
		environment,
	);

	log(
		`Using training options: ${JSON.stringify(
			trainingOptions,
			undefined,
			"\t",
		)}`,
	);
	log(`Using environment ${environment.name}`);
	log(`Score to beat: ${environment.winningScore ?? "[not set]"}`);

	const didWin = train(environment, agent, trainingOptions);
	log(didWin ? "You won!" : "You lost.");

	if (agent.save) {
		await agent.save(options.saveDirectory);
	}
};

main().catch(console.error);
