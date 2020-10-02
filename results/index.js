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
	return new Promise((resolve) => {
		xobj.onreadystatechange = () => {
			if (xobj.readyState === 4 && xobj.status === 200) {
				resolve(JSON.parse(xobj.responseText));
			}
		};
		xobj.send(null);
	});
};

const convertToPoint = (y, x) => ({ x, y });

async function run() {
	const experimentName = getExperimentName(state);
	const filePath = `data/${experimentName}.json`;
	const { returns, rollingAverageReturns } = await loadJSON(filePath);
	const returnsPoints = returns.map(convertToPoint);
	const rollingAverageReturnsPoints = rollingAverageReturns.map(convertToPoint);

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

	state = {
		...state,
		timeout: setTimeout(run, 5000),
	};
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
