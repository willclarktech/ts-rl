/* globals document,XMLHttpRequest,tfvis */

async function loadJSON(path) {
	const xobj = new XMLHttpRequest();
	xobj.overrideMimeType("application/json");
	xobj.open("GET", path, true);
	return new Promise((resolve) => {
		xobj.onreadystatechange = function () {
			if (xobj.readyState == 4 && xobj.status == "200") {
				// Required use of an anonymous callback as .open will NOT return a value but simply returns undefined in asynchronous mode
				resolve(JSON.parse(xobj.responseText));
			}
		};
		xobj.send(null);
	});
}

async function run() {
	const data = await loadJSON("data/xxx.json");
	tfvis.render.scatterplot(
		{ name: "Horsepower v MPG" },
		{ values: data },
		{
			xLabel: "Horsepower",
			yLabel: "MPG",
			height: 300,
		},
	);
}

document.addEventListener("DOMContentLoaded", run);
