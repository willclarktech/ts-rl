import { Environment } from "../environments";
import { Agent } from "./core";

// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface A3COptions {}

export class A3C implements Agent {
	public readonly name: string;

	public constructor(environment: Environment, options: A3COptions) {
		this.name = "A2C";
		throw new Error(
			`A3C not implemented.\nEnvironment: ${environment}\nOptions:${options}`,
		);
	}

	public runEpisode(environment: Environment): number {
		throw new Error(`not implemented\nEnvironment: ${environment}`);
	}
}
