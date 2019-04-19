<template>
	<div class="container my-5">
		<h2 class="font-weight-bold">Linear Regression (Tensorflow.js)</h2>
		<p class="mt-4 mb-5">
			The following is a Stochastic Linear Regression implementation using Tensorflow.js,
			running against a house pricing dataset.

			Your machine is probably now currently training the model as you are reading this.
			It will automatically stop training after reaching some optimal values.

			<br>

			<a href="https://github.com/MicroDroid/tfjs-linear-regression-demo" class="mt-3 text-info" target="_blank">
				<strong>Source here.</strong>
			</a>
		</p>

		<apexchart height="480" type="line" :options="chartOptions" :series="chartSeries"></apexchart>

		<div class="row no-gutters prediction-form mt-2">
			<div class="col-auto d-flex flex-column justify-content-center">
				A
			</div>
			<div class="col-auto mx-2">
				<input type="number" class="form-control form-control-sm bg-transparent" v-model="formArea">
			</div>
			<div class="col d-flex flex-column justify-content-center">
				<span>
					sqft house is going to likely be priced: <strong>{{ formPrediction.toLocaleString(undefined, {maximumFractionDigits: 0}) }} US$</strong>
				</span>
			</div>
		</div>

		<table class="table borderless w-100 mt-5">
			<tr>
				<th>R<sup>2</sup>:</th>
				<td>{{ R2.toFixed(3) }}</td>

				<th>Mean Squared Error:</th>
				<td>
					{{ mse.toLocaleString() }}
					<span class="text-success">
						({{ regression && regression.mseHistory.length >= 2 ? (mse - regression.mseHistory[1]).toLocaleString() : 0 }})
					</span>
				</td>

			</tr>
			<tr>
				<th>Iterations:</th>
				<td>{{ iterations }}</td>

				<th>Iteration cost (avg.):</th>
				<td>{{ (trainingTimes.reduce((acc, val) => acc + val, 0) / trainingTimes.length || 1).toFixed(0) }}ms</td>
			</tr>
			<tr>
				<th>Dataset size*:</th>
				<td>{{ this.features.length }} (training) / {{ this.testingFeatures.length }} (testing)</td>

				<th>Batch size:</th>
				<td>{{ batchSize }}</td>
			</tr>
		</table>

		<p class="text-muted mt-3">
			<small>
				* Testing set is larger than training set in order to keep each training iteration fast enough to not block brower rendering
			</small>
		</p>
	</div>
</template>

<script>
	import VueApexCharts from 'vue-apexcharts';
	import dataset from '../../csv/houses-mini.csv';
	import LinearRegression from '../LinearRegression';

	export default {
		components: {
			'apexchart': VueApexCharts,
		},

		mounted() {
			const trainingSlice = dataset.slice(0, 150);
			const testingSlice = dataset.slice(150, 400);

			this.features = trainingSlice.map(row => [row.sqft_living]);
			this.labels = trainingSlice.map(row => [row.price]);

			this.testingFeatures = testingSlice.map(row => [row.sqft_living]);
			this.testingLabels = testingSlice.map(row => [row.price]);

			this.regression = new LinearRegression(this.features, this.labels, {
				learningRate: 0.005,
				iterations: 1,
				batchSize: this.batchSize,
			});

			const think = () => setTimeout(() => {
				const startDate = Date.now();
				this.regression.train();
				this.trainingTimes.push(Date.now() - startDate);

				this.R2 = this.regression.test(this.testingFeatures, this.testingLabels);
				this.mse = this.regression.mseHistory[0];

				if (this.iterations >= 250)
					return;

				++this.iterations;
				think();
			}, 32);

			think();
		},
		
		data() {
			return {
				batchSize: 50,

				formArea: 500,

				features: [],
				labels: [],

				testingFeatures: [],
				testingLabels: [],

				regression: null,

				R2: 0,
				mse: 0,
				iterations: 0,
				trainingTimes: [],

				chartOptions: {
					chart: {
						animations: {
							enabled: false,
						},
					},

					markers: {
						size: [6, 0]
					},

					tooltip: {
						shared: false,
						intersect: true,
					},

					xaxis: {
						min: 0,

						title: {
							text: 'Living area (sqft)',
						},
					},

					yaxis: {
						min: 0,

						title: {
							text: 'Price (US$)',
						},
					}
				},
			};
		},

		computed: {
			predictions() {
				return this.regression ? this.regression.predict([
					// [0],
					...this.features,
				]) : [];
			},

			formPrediction() {
				return this.regression ? this.regression.predict([
					[parseInt(this.formArea || 0)],
				])[0][0] : 0;
			},

			chartSeries() {
				const houses = [];
				const predictions = [];

				for (let i = 0; i < this.features.length; i++) {
					houses.push({x: this.features[i][0], y: this.labels[i][0]});
					predictions.push({x: this.features[i][0], y: this.predictions[i][0]});
				}

				return [
					{
						name: 'Data points',
						type: 'scatter',
						data: houses,
					},
					{
						name: 'Predictions',
						type: 'line',
						data: predictions,
					}
				];
			},
		}
	};
</script>


<style scoped>
	table {
		table-layout: fixed;
	}

	.prediction-form input {
		width: 96px;
	}
</style>
