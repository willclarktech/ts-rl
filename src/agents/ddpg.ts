import { Environment } from "../environments";
import { Agent } from "./core";

// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface DDPGOptions {}

export class DDPG implements Agent {
	public readonly name: string;

	public constructor(environment: Environment, options: DDPGOptions) {
		this.name = "DDPG";
		throw new Error(
			`DDPG not implemented.\nEnvironment: ${environment}\nOptions:${options}`,
		);
	}

	public runEpisode(environment: Environment): number {
		throw new Error(`not implemented\nEnvironment: ${environment}`);
	}
}
