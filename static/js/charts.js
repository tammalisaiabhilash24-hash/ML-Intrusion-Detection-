function buildBarChart(container, values, labels) {
    if (!container) return;
    container.innerHTML = '';

    const maxValue = Math.max(...values, 1);
    values.forEach(function (value, index) {
        const item = document.createElement('div');
        item.className = 'bar-item';

        const fill = document.createElement('div');
        fill.className = 'bar-fill';
        fill.style.height = `${(value / maxValue) * 100}%`;

        const label = document.createElement('span');
        label.textContent = labels[index] || '';

        item.appendChild(fill);
        item.appendChild(label);
        container.appendChild(item);
    });
}
