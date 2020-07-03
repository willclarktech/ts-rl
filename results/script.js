/* globals document,XMLHttpRequest,tfvis */

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

async function run() {
	const dataDir = "data";
	const experimentName = "cars";
	const fileName = `${dataDir}/${experimentName}.json`;
	const {
		name,
		xLabel,
		yLabel,
		height,
		originalData,
		predictionData,
	} = await loadJSON(fileName);

	tfvis.render.scatterplot(
		{ name },
		{
			values: [originalData, predictionData],
			series: ["original data", "prediction data"],
		},
		{
			xLabel,
			yLabel,
			height,
		},
	);
}

document.addEventListener("DOMContentLoaded", run);
