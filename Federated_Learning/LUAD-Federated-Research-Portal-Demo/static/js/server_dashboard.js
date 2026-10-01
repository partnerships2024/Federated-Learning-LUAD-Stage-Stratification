/* Research visualizations for the admin-only server dashboard. */
(function () {
  "use strict";

  const dashboardData = window.serverDashboardData || {};
  const rounds = dashboardData.rounds || [];
  const comparison = dashboardData.comparison || [];
  const fontFamily = "'IBM Plex Sans', 'Segoe UI', sans-serif";
  const axisColor = "#536c7d";
  const gridColor = "rgba(16, 42, 67, 0.10)";

  if (typeof Chart === "undefined") {
    return;
  }

  const legendOptions = {
    position: "top",
    labels: {
      color: "#1b3347",
      boxWidth: 14,
      boxHeight: 10,
      padding: 18,
      font: { family: fontFamily, size: 13, weight: "600" },
    },
  };

  const scaleTitle = (text) => ({
    display: true,
    text,
    color: axisColor,
    font: { family: fontFamily, size: 12, weight: "600" },
    padding: { top: 8, bottom: 3 },
  });

  const tickOptions = {
    color: axisColor,
    font: { family: fontFamily, size: 12, weight: "500" },
    padding: 5,
  };

  const roundCanvas = document.getElementById("roundPerformanceChart");
  if (roundCanvas && rounds.length) {
    new Chart(roundCanvas, {
      type: "line",
      data: {
        labels: rounds.map((item) => `Round ${item.round}`),
        datasets: [
          {
            label: "Weighted validation ROC-AUC",
            data: rounds.map((item) => item.roc_auc),
            borderColor: "#168b8d",
            backgroundColor: "#168b8d",
            pointBackgroundColor: "#168b8d",
            pointBorderColor: "#ffffff",
            pointBorderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 5,
            borderWidth: 2,
            tension: 0.25,
            fill: false,
          },
          {
            label: "Weighted validation F1",
            data: rounds.map((item) => item.f1),
            borderColor: "#536c7d",
            backgroundColor: "#536c7d",
            pointBackgroundColor: "#536c7d",
            pointBorderColor: "#ffffff",
            pointBorderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 5,
            borderWidth: 2,
            tension: 0.25,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: legendOptions,
          title: { display: false },
        },
        scales: {
          x: {
            ticks: tickOptions,
            grid: { color: gridColor },
            title: scaleTitle("Federated round"),
          },
          y: {
            min: 0,
            max: 1,
            ticks: tickOptions,
            grid: { color: gridColor },
            title: scaleTitle("Metric score"),
          },
        },
      },
    });
  }

  const comparisonCanvas = document.getElementById("comparisonChart");
  if (comparisonCanvas && comparison.length) {
    new Chart(comparisonCanvas, {
      type: "bar",
      data: {
        labels: comparison.map((item) => item.metric),
        datasets: [
          {
            label: "Matched centralized baseline",
            data: comparison.map((item) => item.centralized),
            backgroundColor: "#788b9a",
            borderColor: "#536c7d",
            borderWidth: 1,
            maxBarThickness: 42,
          },
          {
            label: "3-client FedAvg federated model",
            data: comparison.map((item) => item.federated),
            backgroundColor: "#168b8d",
            borderColor: "#0e7073",
            borderWidth: 1,
            maxBarThickness: 42,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: legendOptions,
          title: { display: false },
        },
        scales: {
          x: {
            ticks: tickOptions,
            grid: { display: false },
            title: scaleTitle("Evaluation metric"),
          },
          y: {
            min: 0,
            max: 1,
            ticks: tickOptions,
            grid: { color: gridColor },
            title: scaleTitle("Metric score"),
          },
        },
      },
    });
  }
})();
