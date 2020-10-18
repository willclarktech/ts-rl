import { Agent } from "./agents";
import { Environment } from "./environments";
import { TrainingOptions } from "./options";
import { log, logEpisode, mean } from "./util";

export const train = (
	environment: Environment,
	agent: Agent,
	{
		maxEpisodes,
		rollingAveragePeriod,
		logPeriod,
		logDirectory,
		warmupEpisodes,
	}: TrainingOptions,
): boolean => {
	const logFile = `${logDirectory}/${agent.name}-${environment.name}.json`;

	let returns: readonly number[] = [];
	let rollingAverageReturns: readonly number[] = [];

	log(`Warming up for ${warmupEpisodes} episodes...`);
	for (let episode = 1; episode <= warmupEpisodes; ++episode) {
		agent.runEpisode(environment, true);
	}
	log("Finished warming up");

	for (let episode = 1; episode <= maxEpisodes; ++episode) {
		const ret = agent.runEpisode(environment);
		returns = [...returns, ret];

		const rollingAverageReturn = mean(returns.slice(-100));
		rollingAverageReturns = [...rollingAverageReturns, rollingAverageReturn];

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
