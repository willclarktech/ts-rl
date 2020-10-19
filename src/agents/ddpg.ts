import { Environment } from "../environments";
import { Agent } from "./core";

export interface DDPGOptions {}

export class DDPG implements Agent {
	public readonly name: string;

	public constructor(environment: Environment, options: DDPGOptions) {
		this.name = "DDPG";
	}

	public runEpisode(environment: Environment): number {
		throw new Error("not implemented");
	}
}
