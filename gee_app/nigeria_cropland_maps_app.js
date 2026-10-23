// Google Earth Engine app to visualize the Nigeria 2020 cropland maps (v4) side-by-side with ESA WorldCover 2020.
// Published at: https://joaquingajardocastillo.users.earthengine.app/view/nigeria-cropland-maps
// Source: users/joaquingajardocastillo/scripts:Papers/Gajardo2023_nigeria_cropland_map_viz_script_v2
// To run it, paste it into the GEE Code Editor (https://code.earthengine.google.com/). The map assets it loads
// (projects/ee-testjoaquin/assets/...) must be readable by your account, or replaced by your own upload of the maps.

var prob_vis = ui.import && ui.import("prob_vis", "imageVisParam", {
      "params": {
        "min": 0,
        "max": 1,
        "palette": [
          "#000004",
          "#2C105C",
          "#711F81",
          "#B63679",
          "#EE605E",
          "#FDAE78",
          "#FCFDBF"
        ]
      }
    }) || {"min":0,"max":1,"palette":["#000004","#2C105C","#711F81","#B63679","#EE605E","#FDAE78","#FCFDBF"]},
    binary_vis = ui.import && ui.import("binary_vis", "imageVisParam", {
      "params": {
        "opacity": 1,
        "bands": [
          "b1"
        ],
        "max": 1,
        "palette": [
          "e6d186",
          "2eaf3e"
        ]
      }
    }) || {"opacity":1,"bands":["b1"],"max":1,"palette":["e6d186","2eaf3e"]},
    countries = ui.import && ui.import("countries", "table", {
      "id": "FAO/GAUL/2015/level0"
    }) || ee.FeatureCollection("FAO/GAUL/2015/level0"),
    nigeria_cropland_binary = ui.import && ui.import("nigeria_cropland_binary", "image", {
      "id": "projects/ee-testjoaquin/assets/Papers-output/Gajardo2023_cropland_Nigeria2020_v4_combined_binary_uint8_clipped_lzw"
    }) || ee.Image("projects/ee-testjoaquin/assets/Papers-output/Gajardo2023_cropland_Nigeria2020_v4_combined_binary_uint8_clipped_lzw"),
    nigeria_cropland_probability = ui.import && ui.import("nigeria_cropland_probability", "image", {
      "id": "projects/ee-testjoaquin/assets/Papers-output/Gajardo2023_cropland_Nigeria2020_v4_combined_probability_uint8_clipped_lzw"
    }) || ee.Image("projects/ee-testjoaquin/assets/Papers-output/Gajardo2023_cropland_Nigeria2020_v4_combined_probability_uint8_clipped_lzw");

// Script inspired by: https://gee-community-catalog.org/projects/harvest/

/*** Maps settings ***/
var map_left = ui.Map()
map_left.setOptions('SATELLITE')
map_left.setCenter({
  lon: ui.url.get('lon', 8.0),
  lat: ui.url.get('lat', 8.0),
  zoom: ui.url.get('zoom', 5.5)
});

var map_right = ui.Map()
map_right.setOptions('SATELLITE')
map_right.setCenter({
  lon: ui.url.get('lon', 8.0),
  lat: ui.url.get('lat', 8.0),
  zoom: ui.url.get('zoom', 5.5)
});

// ESA landcover map
var WorldCover = ee.ImageCollection('ESA/WorldCover/v100').first();
var nigeria_borders = countries.filter(ee.Filter.eq('ADM0_NAME','Nigeria'));
var worldcover_vis = {bands: ['Map']};
map_left.addLayer(WorldCover.clip(nigeria_borders), worldcover_vis, 'ESA WorldCover 2020');

// Nigeria maps
var nigeria_cropland_probability = nigeria_cropland_probability.divide(100); // Scale nigeria data between 0 and 1
map_right.addLayer(nigeria_cropland_binary, binary_vis, 'Nigeria Cropland Binary (0.5 threshold)');
map_right.addLayer(nigeria_cropland_probability, prob_vis, 'Nigeria Cropland Probability');

/*** Legend Cropland binary map ***/

var croplandGFSAD_vis = {min: 0.0, max: 1.0, palette: binary_vis.palette};
var croplandlegendPanel = ui.Panel({
  style: {
    position: 'bottom-right',
    height: '75px',
    padding: '5px',
    backgroundColor: 'rgba(255, 255, 255, 0.00001)',
  }
});
// Create legend title
var croplandlegendTitle = ui.Label({
  value: 'Cropland binary (0.5 threshold)',
  style: {
    fontWeight: 'bold',
    color: 'white',
    margin: '0 0 4px 0',
    padding: '0',
    fontSize: '14px',
    backgroundColor: 'rgba(255, 255, 255, 0.00001)',
    }
});
// Add the title to the panel
croplandlegendPanel.add(croplandlegendTitle);   

//Creates and styles 1 row for the cropland legend.
var makeRow = function(color, name) {
      // Create the label that is actually the colored box.
      var colorBox = ui.Label({
        style: {
          backgroundColor: color,
          // Use padding to give the box height and width.
          padding: '8px',
          margin: '0 0 4px 0', 
        }
      });
      // Create the label filled with the description text.
      var description = ui.Label({
        value: name,
        style: {
          margin: '0 0 2px 4px', 
          fontSize: '12px',
          color: 'white',
          backgroundColor: 'rgba(255, 255, 255, 0.00001)',
        }
      });
      // return the panel
      return ui.Panel({
        widgets: [colorBox, description],
        layout: ui.Panel.Layout.Flow('horizontal'),
        style: {backgroundColor: 'rgba(255, 255, 255, 0.00001)',}
      });
};

//Add color and and legend names
var names = ['Non-cropland', 'Cropland'];
for (var i = 1; i >=0; i--) {
  croplandlegendPanel.add(makeRow(croplandGFSAD_vis.palette[i], names[i]));
  }
  
map_right.add(croplandlegendPanel);

/*** Legend Cropland Probability map ***/
// Credits to Justin Braaten: https://gis.stackexchange.com/questions/307652/how-to-add-color-map-scale-in-google-earth-engine
function makeColorBarParams(palette) {
  return {
    bbox: [0, 0, 1, 0.1], //this is the length of the bar, same for all colorbars
    dimensions: '100x10',
    format: 'png',
    min: 0,
    max: 1,
    palette: palette,
  };
}
// Create the color bar for the legend.
var colorBar = ui.Thumbnail({
    image: ee.Image.pixelLonLat().select(0),
    params: makeColorBarParams(prob_vis.palette),
    style: {stretch: 'horizontal', margin: '0px 0px', maxHeight: '5px', position: 'bottom-left', backgroundColor: 'rgba(255, 255, 255, 0.00001)'},
  });
  
// Create a panel with three numbers for the legend.
function makeColorBarLegend(vis) {
  var barMin = vis.min;
  var barMiddle = (vis.max-vis.min) / 2 + vis.min;
  var barMax = vis.max;
  if (barMax !== ee.Number(barMax).toInt().getInfo()) { // only display float if necessary
    barMin = barMin.toFixed(2);
    barMiddle = barMiddle.toFixed(2);
    barMax = barMax.toFixed(2);
  }
  return ui.Panel({
    widgets: [
      ui.Label(barMin, {margin: '4px 2px', fontSize: '10px', color: 'white', backgroundColor: 'rgba(255, 255, 255, 0.00001)'}),
      ui.Label(barMiddle, {margin: '4px 0px', textAlign: 'center', stretch: 'horizontal', fontSize: '10px', color: 'white', backgroundColor: 'rgba(255, 255, 255, 0.00001)'
      }),
      ui.Label(barMax, {margin: '4px 0px', fontSize: '10px', color: 'white', backgroundColor: 'rgba(255, 255, 255, 0.00001)'})
    ],
    layout: ui.Panel.Layout.flow('horizontal'), // makes number in colorbar aligned horizontally
    style: {
      backgroundColor: 'rgba(255, 255, 255, 0.01)',
      //padding: '0px',
     // margin: '0px'
    }
  });
}
var legendTitle = ui.Label({value: 'Cropland probability', style: {color: 'white', fontWeight: 'bold', fontSize: '14px', backgroundColor: 'rgba(255, 255, 255, 0.00001)'}});

// Add the legendPanel to the map
var croplandProblegendPanel = ui.Panel(
                  [legendTitle, colorBar, makeColorBarLegend(prob_vis)],
                  'flow',
                  {
                    position: 'bottom-right',
                    backgroundColor: 'rgba(255, 255, 255, 0.00001)',
                    minWidth: '210px',
                  }
);
map_right.add(croplandProblegendPanel);


/*** Legend ESA WorldCover landcover map ***/
var classes_dict = {
  10	: 'Tree cover',
  20	: 'Shrubland',
  30	: 'Grassland',
  40	: 'Cropland',
  50	: 'Built-up',
  60	: 'Bare / sparse vegetation',
  70	: 'Snow and ice',
  80	: 'Permanent water bodies',
  90	: 'Herbaceous wetland',
  95	: 'Mangroves',
  100	: 'Moss and lichen',
};

var landcover_palette = [
          '#006400',	
          '#ffbb22',	
          '#ffff4c',	
          '#f096ff',	
          '#fa0000',	
          '#b4b4b4',	
          '#f0f0f0',	
          '#0064c8',	
          '#0096a0',	
          '#00cf75',	
          '#fae6a0',
];

var classes_names = [
          'Tree cover',
          'Shrubland',
          'Grassland',
          'Cropland',
          'Built-up',
          'Bare / sparse vegetation',
          'Snow and ice',
          'Permanent water bodies',
          'Herbaceous wetland',
          'Mangroves',
          'Moss and lichen',
];

var makeRow = function(color, name) {
      // Create the label that is actually the colored box.
      var colorBox = ui.Label({
        style: {
          backgroundColor: color,
          // Use padding to give the box height and width.
          padding: '6px',
          margin: '0 0 4px 0', 
        }
      });
      // Create the label filled with the description text.
      var description = ui.Label({
        value: name,
        style: {
          margin: '0 0 2px 4px', 
          fontSize: '9px',
        }
      });
      // return the panel
      return ui.Panel({
        widgets: [colorBox, description],
        layout: ui.Panel.Layout.Flow('horizontal')
      });
};

var landcoverlegendPanel = ui.Panel({style: {
    position: 'bottom-left',
    padding: '4px 10px'
  }});
  
// Create legend title  
var landcoverlegendTitle = ui.Label({
  value: 'ESA WorldCover 2020',
  style: {
    fontWeight: '700',
    margin: '0 0 4px 0',
    padding: '0',
    fontSize: '11px',
    color: '#383838'
    }
});
landcoverlegendPanel.add(landcoverlegendTitle);

// Add color and and legend names
var keys = Object.keys(classes_dict);
for (var i = 0; i <=10; i++) {
  landcoverlegendPanel.add(makeRow(landcover_palette[i], classes_names[i]));
  }  

map_left.add(landcoverlegendPanel);


// Splitter
var linker = ui.Map.Linker([map_left, map_right]);
var left_panel = ui.Panel([map_left], null, {stretch: 'both'});
var right_panel = ui.Panel([map_right], null, {stretch: 'both'});
var split = ui.SplitPanel(left_panel, right_panel, 'horizontal', true);

ui.root.widgets().reset([split]);
ui.root.setLayout(ui.Panel.Layout.Flow('vertical'));


