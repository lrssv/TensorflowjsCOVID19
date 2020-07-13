/*$("#image-selector").change(function () {
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#selected-image").attr("src", dataURL);
		$("#prediction-list").empty();
	}
	
	let file = $("#image-selector").prop('files')[0];
	reader.readAsDataURL(file);
});

let model;
$( document ).ready(async function () {
	$('.progress-bar').show();
	console.log( "Loading model..." );
*/

/* IMPORTAR O MODELO DE REDES NEURAIS PARA O TENSORFLOW JS*/
	model = await tf.loadGraphModel('model/model.json');

/*	
    console.log( "Model loaded." );
	$('.progress-bar').hide();
});

$("#predict-button").click(async function () {
	let image = $('#selected-image').get(0);
	
	// Pre-process the image
	console.log( "Loading image..." );
*/	

/* CRIAR A VARIAVEL COM A IMAGEM PARA ALIMENTAR A REDE*/ 	
	let tensor = tf.browser.fromPixels(image)
		.resizeNearestNeighbor([300, 300]) // change the image size
		.expandDims()
		.toFloat()

/* VARIAVEL QUE FAZ A PREVISÃO */		
	let predictions = await model.predict(tensor.div(255));
	
/* ARMAZENAR DOIS ARRAYS COM O DIAGNÓSTICO E COM A PROBABILIDADE */	
	const {values, indices} = tf.topk(predictions,3);
	
	/*ARRAY DE PROBABILIDADES*/	
	const classProbs = values.arraySync();

	/*ARRAY DE DIAGNÓSTICO*/
    const classIndices = indices.arraySync();

	/*IMPRIME RESULTADO*/
	console.log(classProbs);
	console.log(classIndices);

/*
	let top5 = Array.from(classIndices)
		.map(function (elem, i) { // this is Array.map
			return {
				className: TARGET_CLASSES[i] // we are selecting the value from the obj
			};
		}).slice(0,3)

	top5.forEach(function (p){
		console.log(p)
	})
		
	
	$("#prediction-list").empty();
	classProbs.forEach(function (p) {
		$("#prediction-list").append(`<li>${p.className}: ${p}</li>`);
		});
});
*/