import { prisma } from "./config/prisma.js";

const metrics = [
  {
    key: "median_after_tax_income",
    displayName: "Median After-Tax Household Income"
  },
  {
    key: "low_income_rate",
    displayName: "Low Income Rate"
  },
  {
    key: "employment_rate",
    displayName: "Employment Rate"
  },
  {
    key: "unemployment_rate",
    displayName: "Unemployment Rate"
  },
  {
    key: "shelter_cost_ratio",
    displayName: "Shelter Cost To Income Ratio"
  },
  {
    key: "homeownership_rate",
    displayName: "Homeownership Rate"
  },
  {
    key: "post_secondary_education",
    displayName: "Post Secondary Education"
  },
  {
    key: "life_expectancy",
    displayName: "Life Expectancy"
  },
  {
    key: "physicians_per_1000",
    displayName: "Physicians Per 1000"
  },
  {
    key: "consumer_price_index",
    displayName: "Consumer Price Index"
  }
];

async function main() {
  for (const metric of metrics) {
    await prisma.metric.upsert({
      where: { key: metric.key },
      update: {},
      create: {
        ...metric,
        source: "Statistics Canada"
      }
    });
  }

  console.log("Metrics seeded");
}

main()
  .catch(console.error)
  .finally(() => prisma.$disconnect());