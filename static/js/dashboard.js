document.addEventListener('DOMContentLoaded', function () {
    const charts = document.querySelectorAll('.bar-chart');
    charts.forEach(function (chart) {
        const values = JSON.parse(chart.dataset.values || '[]');
        const labels = JSON.parse(chart.dataset.labels || '[]');
        buildBarChart(chart, values, labels);
    });
});
