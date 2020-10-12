/* globals document, XMLHttpRequest, tfvis */

const algorithms = ["DQN", "Random", "Reinforce"];
const environments = ["Blackjack", "CartPole"];

let state = {
	algorithm: "Random",
	environment: "Blackjack",
	timeout: null,
};

const getExperimentName = ({ algorithm, environment }) =>
	`${algorithm}-${environment}`;

const loadJSON = async (path) => {
	const xobj = new XMLHttpRequest();
	xobj.overrideMimeType("application/json");
	xobj.open("GET", path, true);
	return new Promise((resolve, reject) => {
		xobj.onreadystatechange = () => {
			if (xobj.readyState === 4) {
				if (xobj.status === 200) {
					resolve(JSON.parse(xobj.responseText));
				} else if (xobj.status >= 400) {
					reject("Error loading JSON");
				}
			}
		};
		xobj.send(null);
	});
};

const convertToPoint = (y, x) => ({ x, y });

const setError = (error) =>
	(document.getElementById("error").innerText = error || "");

async function run() {
	setError();
	const experimentName = getExperimentName(state);
	const filePath = `data/${experimentName}.json`;
	try {
		const { returns, rollingAverageReturns } = await loadJSON(filePath);
		const returnsPoints = returns.map(convertToPoint);
		const rollingAverageReturnsPoints = rollingAverageReturns.map(
			convertToPoint,
		);

		tfvis.visor().open();
		tfvis.render.linechart(
			{ name: "visor" },
			{
				values: [returnsPoints, rollingAverageReturnsPoints],
				series: ["returns", "rolling average return (100 episodes)"],
			},
			{
				xLabel: "episode",
				yLabel: "return",
				height: 500,
			},
		);

		if (state.timeout) {
			clearTimeout(state.timeout);
		}
		state = {
			...state,
			timeout: setTimeout(run, 5000),
		};
	} catch (error) {
		setError(error);
	}
}

const onLoad = () => {
	document.getElementById("algorithm-name").innerHTML = algorithms.map(
		(algorithm) =>
			`<option value=${algorithm} ${
				algorithm === state.algorithm ? "selected" : ""
			}>${algorithm}</option>`,
	);
	document.getElementById("environment-name").innerHTML = environments.map(
		(environment) =>
			`<option value="${environment}" ${
				environment === state.environment ? "selected" : ""
			}>${environment}</option>`,
	);
	document.getElementById("environment-form").onsubmit = (event) => {
		event.preventDefault();
		state = {
			algorithm: document.getElementById("algorithm-name").value,
			environment: document.getElementById("environment-name").value,
			timeout: state.timeout || clearTimeout(state.timeout),
		};
		run();
	};
};

document.addEventListener("DOMContentLoaded", onLoad);
