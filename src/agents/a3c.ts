import { Environment } from "../environments";
import { Agent } from "./core";

export interface A3COptions {}

export class A3C implements Agent {
	public readonly name: string;

	public constructor(environment: Environment, options: A3COptions) {
		this.name = "A2C";
	}

	public runEpisode(environment: Environment): number {
		throw new Error("not implemented");
	}
}
