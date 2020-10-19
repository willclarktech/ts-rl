import { A3C, ActorCritic, Agent, DQN, Random, Reinforce } from "./agents";
import { Blackjack, CartPole, Environment, MountainCar } from "./environments";
import * as options from "./options";
import { setSeed } from "./random";
import { train } from "./train";
import { log } from "./util";

const createEnvironment = (environmentName: string): Environment => {
	switch (environmentName) {
		case "Blackjack":
			return new Blackjack();
		case "CartPole":
			return new CartPole();
		case "MountainCar":
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
		case "A3C": {
			const a3cOptions = options.A3C[environment.name];
			verifyOptions(a3cOptions, agentName, environment.name);
			const { trainingOptions, ...agentOptions } = a3cOptions;
			return {
				agent: new A3C(environment, agentOptions),
				trainingOptions,
			};
		}
		case "ActorCritic": {
			const actorCriticOptions = options.ActorCritic[environment.name];
			verifyOptions(actorCriticOptions, agentName, environment.name);
			const { trainingOptions, ...agentOptions } = actorCriticOptions;
			return {
				agent: new ActorCritic(environment, agentOptions),
				trainingOptions,
			};
		}
		case "DQN": {
			const dqnOptions = options.DQN[environment.name];
			verifyOptions(dqnOptions, agentName, environment.name);
			const { trainingOptions, ...agentOptions } = dqnOptions;
			return {
				agent: new DQN(environment, agentOptions),
				trainingOptions,
			};
		}
		case "Random": {
			const randomOptions = options.Random[environment.name];
			verifyOptions(randomOptions, agentName, environment.name);
			const { trainingOptions } = randomOptions;
			return {
				agent: new Random(environment),
				trainingOptions,
			};
		}
		case "Reinforce": {
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
	const agentName = process.argv[2];
	const environmentName = process.argv[3];

	if (!agentName || !environmentName) {
		console.error("Usage: npm start <agent name> <environment name>");
		console.error("Example: npm start ActorCritic CartPole");
		process.exit(1);
	}

	try {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const { seed } = (options as any)[agentName][
			environmentName
		].trainingOptions;
		setSeed(seed);
	} catch (error) {
		// seed not specified
	}

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
