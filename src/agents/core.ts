import { Environment } from "../environments";

export interface Agent {
	name: string;
	runEpisode(env: Environment, warmup?: boolean): number;
	save?(directory: string): Promise<void>;
}
