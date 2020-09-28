/* globals document, XMLHttpRequest, tfvis */

let currentExperimentName = "CartPole-Reinforce";
let timeout = null;

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

async function run(experimentName) {
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

	document.getElementById("environment").onchange = (event) => {
		currentExperimentName = `${event.target.value}-Reinforce`;
		if (timeout) {
			clearTimeout(timeout);
		}
		run(currentExperimentName);
	};
	timeout = setTimeout(run.bind(null, currentExperimentName), 5000);
}

document.addEventListener(
	"DOMContentLoaded",
	run.bind(null, currentExperimentName),
);
