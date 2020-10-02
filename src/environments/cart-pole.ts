import { Environment, Observation, Sample } from "./core";

export class CartPole implements Environment {
	public readonly name: string;
	public readonly winningScore: number;
	public readonly numObservationDimensions: number;
	public readonly numActions: number;

	private readonly gravity: number;
	private readonly massCart: number;
	private readonly massPole: number;
	private readonly totalMass: number;
	private readonly length: number;
	private readonly poleMoment: number;
	private readonly forceMagnitude: number;
	private readonly tau: number;
	private readonly thetaThresholdRadians: number;
	private readonly xThreshold: number;
	private readonly maxEpisodeLength: number;

	private steps: number;
	private done: boolean;
	private state: readonly [number, number, number, number];

	public constructor() {
		this.name = "CartPole";
		this.winningScore = 195;
		this.numObservationDimensions = 4;
		this.numActions = 2;

		this.gravity = 9.8;
		this.massCart = 1.0;
		this.massPole = 0.1;
		this.totalMass = this.massPole + this.massCart;
		this.length = 0.5; // to centre of mass
		this.poleMoment = this.massPole * this.length;
		this.forceMagnitude = 10.0;
		this.tau = 0.02; // seconds between state updates

		// angle at which to fail the episode
		this.thetaThresholdRadians = (12 * 2 * Math.PI) / 360;
		this.xThreshold = 2.4;

		this.maxEpisodeLength = 200;

		this.steps = 0;
		this.done = true;
		this.state = [0, 0, 0, 0];
	}

	public reset(): Observation {
		const x = Math.random() - 0.5; // cart position
		const xDot = Math.random() - 0.5; // cart velocity
		const theta = (Math.random() - 0.5) * 2 * ((6 / 360) * 2 * Math.PI); // pole angle (radians)
		const thetaDot = (Math.random() - 0.5) * 0.5; // pole anguular velocity

		this.steps = 0;
		this.state = [x, xDot, theta, thetaDot];
		this.done = false;
		return this.state;
	}

	public step(action: number): Sample {
		if (action >= this.numActions) {
			throw new Error("Action is not in range");
		}
		if (this.done) {
			throw new Error("Env is done");
		}

		const [x, xDot, theta, thetaDot] = this.state;
		const force = action === 1 ? this.forceMagnitude : -this.forceMagnitude;
		const cosTheta = Math.cos(theta);
		const sinTheta = Math.sin(theta);

		const temperature =
			(force + this.poleMoment * thetaDot ** 2 * sinTheta) / this.totalMass;
		const thetaAcceleration =
			(this.gravity * sinTheta - cosTheta * temperature) /
			(this.length *
				(4.0 / 3.0 - (this.massPole * cosTheta ** 2) / this.totalMass));
		const xAcceleration =
			temperature -
			(this.poleMoment * thetaAcceleration * cosTheta) / this.totalMass;

		const xNext = x + this.tau * xDot;
		const xDotNext = xDot + this.tau * xAcceleration;
		const thetaNext = theta + this.tau * thetaDot;
		const thetaDotNext = thetaDot + this.tau * thetaAcceleration;

		this.steps += 1;
		this.state = [xNext, xDotNext, thetaNext, thetaDotNext];
		this.done =
			this.steps >= this.maxEpisodeLength ||
			xNext < -this.xThreshold ||
			xNext > this.xThreshold ||
			thetaNext < -this.thetaThresholdRadians ||
			thetaNext > this.thetaThresholdRadians;

		return {
			observation: this.state,
			reward: 1,
			done: this.done,
		};
	}
}
