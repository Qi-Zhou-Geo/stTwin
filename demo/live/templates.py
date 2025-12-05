# templates.py
# Store all HTML templates here

INDEX_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>stTwin</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .info {
            text-align: center;
            margin: 20px 0;
            color: #666;
        }
        .status {
            display: inline-block;
            padding: 8px 16px;
            background: #4CAF50;
            color: white;
            border-radius: 20px;
            font-size: 14px;
        }
        .plots-container {
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
            margin-top: 30px;
        }
        .plot-section {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background: #f9f9f9;
        }
        .plot-title {
            font-size: 16px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            text-align: center;
        }
        .plot {
            width: 100%;
            height: 350px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>stTwin [Station: MVE]</h1>
        <div class="info">
            <p>Last Update: <span id="lastUpdate">Loading...</span> (Auto-refresh every 10 minutes)</p>
        </div>

        <div class="plots-container">
            <div class="plot-section">
                <div class="plot-title">Precipitation [mm per 10 minutes]</div>
                <div id="plot1" class="plot"></div>
            </div>
            <div class="plot-section">
                <div class="plot-title">Temperature [degree]</div>
                <div id="plot2" class="plot"></div>
            </div>
            <div class="plot-section">
                <div class="plot-title">Sun Radiation [W per squared m]</div>
                <div id="plot3" class="plot"></div>
            </div>
        </div>
    </div>

    <script>
        function updatePlot() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    console.log("Data received:", data);
                    document.getElementById('lastUpdate').textContent = data.timestamp || 'N/A';

                    if (data.data && data.data.length > 0) {
                        const timestamps = data.data.map(row => row['timestamp [UTC+0]']);

                        const columnsToPlot = [
                            {id: 'plot1', column: 'precipitation [mm per 10 minutes]', color: 'blue'},
                            {id: 'plot2', column: 'temperature [degree]', color: 'red'},
                            {id: 'plot3', column: 'sun radiation [W per squared m]', color: 'orange'}
                        ];

                        columnsToPlot.forEach(plot => {
                            const values = data.data.map(row => {
                                const val = row[plot.column];
                                // Handle NaN values
                                return (val === null || val === undefined || isNaN(val)) ? null : val;
                            });

                            const trace = {
                                x: timestamps,
                                y: values,
                                type: 'scatter',
                                mode: 'lines+markers',
                                line: {color: plot.color, width: 2},
                                marker: {size: 4}
                            };

                            const layout = {
                                margin: {l: 60, r: 40, t: 30, b: 50},
                                xaxis: {
                                    title: 'Time [UTC+0]'
                                },
                                yaxis: {
                                    title: plot.column
                                },
                                hovermode: 'closest'
                            };

                            Plotly.newPlot(plot.id, [trace], layout, {responsive: true});
                        });
                    }
                })
                .catch(error => console.error('Error fetching data:', error));
        }

        // Initial plot
        updatePlot();

        // Update every 10 minutes (600000 ms)
        setInterval(updatePlot, 600000);

        // Also update every 5 seconds for the first minute to catch initial data
        let quickUpdateCount = 0;
        const quickUpdate = setInterval(() => {
            updatePlot();
            quickUpdateCount++;
            if (quickUpdateCount >= 12) {
                clearInterval(quickUpdate);
            }
        }, 5000);
    </script>
</body>
</html>

"""