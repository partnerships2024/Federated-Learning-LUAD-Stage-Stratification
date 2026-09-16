/* Research visualizations for the admin-only server dashboard. */
(function () {
  "use strict";

  const dashboardData = window.serverDashboardData || {};
  const rounds = dashboardData.rounds || [];
  const comparison = dashboardData.comparison || [];

  if (typeof Chart === "undefined") {
    return;
  }

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
            borderColor: "#2166d1",
            backgroundColor: "rgba(33, 102, 209, 0.12)",
            tension: 0.25,
            fill: true,
          },
          {
            label: "Weighted validation F1",
            data: rounds.map((item) => item.f1),
            borderColor: "#159a9c",
            backgroundColor: "rgba(21, 154, 156, 0.08)",
            tension: 0.25,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { title: { display: true, text: "Federated Performance Across Communication Rounds" } },
        scales: {
          x: { title: { display: true, text: "Federated Round" } },
          y: { min: 0, max: 1, title: { display: true, text: "Metric Score" } },
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
            backgroundColor: "#159a9c",
          },
          {
            label: "3-client FedAvg federated model",
            data: comparison.map((item) => item.federated),
            backgroundColor: "#2166d1",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { title: { display: true, text: "Matched Centralized vs Federated Performance" } },
        scales: { y: { min: 0, max: 1, title: { display: true, text: "Metric Score" } } },
      },
    });
  }
})();
