import { prisma } from "./config/prisma.js";
import { provinces } from "./utils/provinces.js";

async function main() {
  for (const [name, code] of Object.entries(provinces)) {
    await prisma.province.upsert({
      where: { code },
      update: {},
      create: {
        code,
        name
      }
    });
  }

  console.log("Provinces seeded");
}

main()
  .catch(console.error)
  .finally(() => prisma.$disconnect());