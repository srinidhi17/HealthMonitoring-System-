
// set the dimensions and margins of the graph
var margin = {top: 30, right: 30, bottom: 100, left: 60},
    width = 1200 - margin.left - margin.right,
    height = 600 - margin.top - margin.bottom;

  
var q =d3.queue()
q.defer(d3.json,"/pymongo_test/posts");       
q.await(makeGraphs);
//console.log(makeGraphs);

function makeGraphs(error,projectsJson){
var data = projectsJson;
//console.log(projectsJson);
//var newdata = "";
//initially new data is set to null.if user deletes a item in the bar chart then new data
// is generated  
var parseDate  = d3.timeParse("%d-%m-%Y %H:%M:%S");
var formatTime = d3.timeFormat("%d-%m-%Y %H:%M");
//drawdata();
var updateInput = function(){

	//typeSelected1 = document.querySelector('input[name=birthdaytime]', '#birthdaytime').value; 
	var parsed = new Date(document.querySelector('input[name=birthdaytime]', '#birthdaytime').value);
  	var ten    = function(x) { return x < 10 ? '00'+x : x};
  	var date   =  parsed.getDate() + '-'+ (parsed.getMonth() + 1) + '-' + parsed.getFullYear();
  	var time   = ten( parsed.getHours() ) + ':' + ten( parsed.getMinutes() );
	startDate = date + ' ' +time;
  	//console.log(startDate);
	var parsed1 = new Date(document.querySelector('input[name=birthdaytime2]', '#birthdaytime').value);
  	var ten1    = function(x) { return x < 10 ? '00'+x : x};
  	var date1  =  parsed1.getDate() + '-'+ (parsed1.getMonth() + 1) + '-' + parsed1.getFullYear();
  	var time1   = ten1( parsed1.getHours() ) + ':' + ten1( parsed1.getMinutes() );
	endDate = date1 + ' ' +time1;
  	console.log('Start Date:',startDate, ' ' ,'End Date:' , endDate);
	//d3.select('#top-5-chart').selectAll('rect').remove();
 
} 

d3.select('#birthdaytime').on('input', updateInput);
var startDate = new Date(document.querySelector('input[name=birthdaytime]', '#birthdaytime').value);
var endDate = new Date(document.querySelector('input[name=birthdaytime2]', '#birthdaytime').value);
console.log('Start Date:',startDate, ' ' ,'End Date:' , endDate);

	// append the svg object to the body of the page
	var chart1 = d3.select("#time-chart")
	  .append("svg")
	    .attr("width", width + margin.left + margin.right)
	    .attr("height", height + margin.top + margin.bottom)
	  .append("g")
	    .attr("transform",
		  "translate(" + margin.left + "," + margin.top + ")")
	


	/*var tip = d3.tip()
	  .attr('class', 'd3-tip')
	  .attr('id','d3-tips')
	  .offset([-10, 0])
	  .html(function(d) {
	    return "<strong> Name  :</strong> <span>" + d.key+ 
	    "</span><br><br> <span > <strong>Count :</strong> " + d.value + 
	    "</span> <span class=\"glyphicon glyphicon-trash\" onclick=\"changedata('"+d.key+"');\"></span>";
	   });
	chart1.call(tip);*/


	// Initialize the X axis
	var x = d3.scaleBand()
	  .range([ 0, width ])
	  .padding(0.4);
	var xAxis = chart1.append("g")
	  .attr("transform", "translate(0," + height + ")")
	   
	// Initialize the Y axis
	var y = d3.scaleLog()
	  .base(2)
	  .domain([1e0,1e5])
	  .range([ height, 0]);
	var yAxis = chart1.append("g")
	  .attr("class", "myYaxis")

	let duplicates = projectsJson;
	projectsJson.forEach(function(d){
		d.Dates = parseDate(d.Dates);
		d.Dates = formatTime(d.Dates);
		if(d.Dates >= "08-01-2021 05:00"){
			//console.log(d.Class,',',d.Dates); 

		}
		 
	}); 
	
	var nest = d3.nest()
	    .key(function(d) { return d.Class; })
	    //.key(function(d){return d.time})
	    .sortKeys(d3.ascending)
	    .rollup(function(leaves) { return leaves.length; })
	    .entries(data)
	    .sort(function(a, b){ return d3.descending(+a.values, +b.values); })
	 console.log(nest);


	 // X axis
	    x.domain(nest.map(function(d) { return d.key; }))
	    xAxis.transition().duration(1000).call(d3.axisBottom(x)).selectAll("text")	
		.style("text-anchor", "end")// End of the X label is rotated
		.attr("dx", "-.8em")
		.attr("dy", ".15em")
		.attr("transform", "rotate(-45)")  
		.style("font-size","14")

	    // Add Y axis
	    y.domain([0.1,100000]);
	    yAxis.transition().duration(1000).call(d3.axisLeft(y));

	    //Create X axis label   
	    chart1.append("text")
		.attr("x", width / 2 )
		.attr("y",  height + margin.bottom )
		.attr("dy", ".01em")
		.style("text-anchor", "middle")
		.style("font-weight","bold")
		.style("font-size","20")
		.text("Classes");


	    // Add a label to the y axis
	    chart1.append("text")
		.attr("transform", "rotate(-45)")
		.attr("y", 0 - 60)
		.attr("x", 0 - (height / 2))
		.attr("dy", "1em")
		.style("font-weight","bold")
		.style("font-size","16")
		.style("text-anchor", "middle")
		.text("Frequency of Occurence")
		.attr("class", "y axis label");


	    // variable u: map data to existing bars
	    var u = chart1.selectAll("rect")
	      .data(nest)
	      .enter()
	      .append("rect")
	      //.merge(chart1)
	      .transition()
	      .duration(1000)
		.attr("x", function(d) { return x(d.key); })
		.attr("y", function(d) { return y(d.value); })// To invert the position for Bar graph plot
		.attr("width", x.bandwidth())
		.attr("fill", "steelblue")
		.attr("height", function(d) { return height - y(d.value); });  
			//.on('mouseover', tip.show);
		
		

		//when user clicks on delete button this function is called
		/*function changedata(item) {

		      $("#d3-tips").remove();
		      $( "#time-graph" ).empty();
		      
		      //the deleted item will be removed from json and bar chart is redrawn with newdata
		      for(var i = newdata.length; i--;) {
			  if(newdata[i].key === item) {
			     newdata.splice(i, 1);
			  }
		      }
		      drawdata();
		}

		//When redraw button is clicked the graph is built again with original data
		$('#redrawbutton').on('click', drawdatamain);

		function drawdatamain()
		{
		 $("#d3-tips").remove();
		$( "#time-graph" ).empty();
		      
		  newdata="";
		  drawdata();

		}



	}*/

var chart2 = d3.select("#top-5-chart")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")")
// Initialize the X axis
var x1 = d3.scaleBand()
  .range([ 0, width ])
  .padding(0.4);
var xAxis1 = chart2.append("g")
  .attr("transform", "translate(0," + height + ")")

// Initialize the Y axis
var y1 = d3.scaleLog()
  .base(2)
  .domain([1e0,1e5])
  .range([ height, 0]);
var yAxis1 = chart2.append("g")
  .attr("class", "myYaxis")


nest.forEach(function (d) {
            d.key = d.key;
            d.value = +d.value;
        });

 var topData = nest.sort(function(a, b) {
    return d3.descending(+a.value, +b.value);
 }).slice(0, 5); //top 10 here
 //console.log(topData);
 //Top 5 Classes
    // X axis
    x1.domain(topData.map(function(d) { return d.key; }))
    xAxis1.transition().duration(1000).call(d3.axisBottom(x1)).selectAll("text")	
        .style("text-anchor", "end")// End of the X label is rotated
        .attr("dx", "-.8em")
        .attr("dy", ".15em")
        .attr("transform", "rotate(-90)")  
	.style("font-size","16")

    // Add Y axis
    y1.domain([0.1,100000]);
    yAxis1.transition().duration(1000).call(d3.axisLeft(y1));


    //Create X axis label   
    chart2.append("text")
	.attr("x", width / 2 )
        .attr("y",  height + margin.bottom )
	.attr("dx", "1em")
        .style("text-anchor", "middle")
        .style("font-weight","bold")
	.style("font-size","20")
        .text("Classes");


    // Add a label to the y axis
    chart2.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 0 - 60)
        .attr("x", 0 - (height / 2))
        .attr("dy", "2em")
	.style("font-weight","bold")
	.style("font-size","16")
        .style("text-anchor", "middle")
        .text("Frequency of Occurence")
        .attr("class", "y axis label");


    // variable u: map data to existing bars
    var v = chart2.selectAll("rect")
      .data(topData)
      .enter()
      .append("rect")
      //.merge(chart1)
      .transition()
      .duration(1000)
        .attr("x", function(d) { return x1(d.key); })
        .attr("y", function(d) { return y1(d.value); })// To invert the position for Bar graph plot
        .attr("width", x1.bandwidth())
        .attr("height", function(d) { return height - y1(d.value); })  
        .attr("fill", "#3289a8");


     var updateRadio = function() {
                typeSelected = document.querySelector('input[name=type-selector]:checked', '#type-selector').value; 
                console.log('typeSelected:', typeSelected);
                //d3.selectAll('rect').remove();
                //drawBar(getDataFromType(typeSelected));
		if(typeSelected == 'Silence'){
		var check = function(d){
			
			if(d.Class == 'Silence'){console.log(d.Class);}
			//if(typeSelected == 'Silence'){if(d.Class == 'Silence'){console.log('Success');}}
			//if(typeSelected == 'Silence'){if(d.Class == 'Silence'){console.log('Success');}}
     		}
		}

		//check;
		
      }

     d3.select('#type-selector').on('change', updateRadio);








};
