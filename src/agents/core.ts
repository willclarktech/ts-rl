import { Environment } from "../environments";

export interface Agent {
	name: string;
	runEpisode(env: Environment): number;
	save?(directory: string): Promise<void>;
}
