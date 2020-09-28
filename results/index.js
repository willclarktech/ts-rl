/* globals window, document, XMLHttpRequest, tfvis */

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
	const experimentName = "CartPole-Reinforce";
	const filePath = `data/${experimentName}.json`;
	const { returns, rollingAverageReturns } = await loadJSON(filePath);
	const returnsPoints = returns.map(convertToPoint);
	const rollingAverageReturnsPoints = rollingAverageReturns.map(convertToPoint);

	tfvis.render.linechart(
		{ name: experimentName },
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

	setTimeout(window.location.reload.bind(window.location), 5000);
}

document.addEventListener("DOMContentLoaded", run);
