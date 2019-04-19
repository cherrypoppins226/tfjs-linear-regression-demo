const tf = require('@tensorflow/tfjs');

export default class LinearRegression {
	constructor(features, labels, options) {
		this.features = tf.tensor(features);
		this.labels = tf.tensor(labels);
		this.mseHistory = [];

		this.features = this.processFeatures(this.features);

		this.options = {
			learningRate: 0.1,
			iterations: 1000,
			batchSize: 10,
			...options,
		};

		this.weights = tf.zeros([this.features.shape[1], 1]);
	}

	gradientDescent(features, labels) {
		const differences = features.matMul(this.weights)
			.sub(labels);


		const slopes = features
			.transpose()
			.matMul(differences)
			.div(features.shape[0]);

		return this.weights.sub(slopes.mul(this.options.learningRate));
	}

	train() {
		const batchQuantity = Math.round(this.features.shape[0] / this.options.batchSize);
		for (let i = 0; i < this.options.iterations; i++) {
			for (let j = 0; j < batchQuantity; j++) {
				const startIndex = j * this.options.batchSize;

				this.weights = tf.tidy(() => {
					const featureSlice = this.features.slice([startIndex, 0], [this.options.batchSize, -1]);
					const labelSlice = this.labels.slice([startIndex, 0], [this.options.batchSize, -1]);

					return this.gradientDescent(featureSlice, labelSlice);
				});
			}

			this.recordMSE();
			// this.updateLearningRate();
		}
	}

	test(testFeatures, testLabels) {
		const {SSres, SStot} = tf.tidy(() => {
			testFeatures = tf.tensor(testFeatures);
			testLabels = tf.tensor(testLabels);

			testFeatures = this.processFeatures(testFeatures);

			const predictions = testFeatures.matMul(this.weights);

			const SSres = testLabels
				.sub(predictions)
				.square()
				.sum()
				.bufferSync()
				.get();

			const SStot = testLabels
				.sub(testLabels.mean())
				.square()
				.sum()
				.bufferSync()
				.get();

			return {SSres, SStot};
		});

		return 1 - (SSres / SStot);
	}

	processFeatures(features) {
		if (this.mean && this.variance)
			features = features.sub(this.mean).div(this.variance.sqrt());
		else
			features = this.standardize(features);

		features = tf.ones([features.shape[0], 1])
			.concat(features, 1);

		return features;
	}

	standardize(features) {
		const {mean, variance} = tf.moments(features, 0);
		
		this.mean = mean;
		this.variance = variance;

		return features.sub(mean).div(variance.sqrt());
	}

	recordMSE() {
		const mse = tf.tidy(() => {
			return this.features
				.matMul(this.weights)
				.sub(this.labels)
				.square()
				.sum()
				.div(this.features.shape[0])
				.bufferSync()
				.get();
		});

		this.mseHistory.unshift(mse);
	}

	updateLearningRate() {
		if (this.mseHistory.length < 2)
			return;

		if (this.mseHistory[0] > this.mseHistory[1])
			this.options.learningRate /= 2;
		else
			this.options.learningRate *= 1.05;
	}

	predict(observations) {
		return tf.tidy(() => {
			observations = tf.tensor(observations);
			observations = this.processFeatures(observations);

			return observations
				.matMul(this.weights)
				.arraySync();
		});
	}
}
