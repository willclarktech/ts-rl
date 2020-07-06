type State = readonly [number, number, number, number];
export type Observation = State;

type Sample = {
	readonly observation: Observation;
	readonly reward: number;
	readonly done: boolean;
};

export class CartPole {
	public static winningScore = 195;

	private state: State;
	private gravity: number;
	private massCart: number;
	private massPole: number;
	private totalMass: number;
	private length: number;
	private poleMoment: number;
	private forceMagnitude: number;
	private tau: number;
	private thetaThresholdRadians: number;
	private xThreshold: number;
	private done: boolean;

	constructor() {
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

		this.done = false;

		this.reset();
	}

	public reset(): Observation {
		const x = Math.random() - 0.5; // cart position
		const xDot = Math.random() - 0.5; // cart velocity
		const theta = (Math.random() - 0.5) * 2 * ((6 / 360) * 2 * Math.PI); // pole angle (radians)
		const thetaDot = (Math.random() - 0.5) * 0.5; // pole anguular velocity

		this.state = [x, xDot, theta, thetaDot];
		this.done = false;
		return this.state;
	}

	public step(action: number): Sample {
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

		this.state = [xNext, xDotNext, thetaNext, thetaDotNext];
		this.done =
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
