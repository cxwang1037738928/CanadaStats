const fs = require('fs');

// Read input file
const input = fs.readFileSync('query.txt', 'utf-8');

// Split into lines, trim, and remove empty lines
const lines = input
  .split('\n')
  .map(line => line.trim())
  .filter(line => line.length > 0);

// Build array of objects with id and query
const result = lines.map((query, index) => ({
  id: index + 1,
  query: query
}));

// Write to query.json
fs.writeFileSync('query.json', JSON.stringify(result, null, 2), 'utf-8');

console.log(`Converted ${result.length} queries into query.json`);