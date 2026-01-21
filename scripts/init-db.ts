
const { initializeDatabase } = require('../src/lib/turso/database');

async function main() {
    console.log('Initializing Database...');
    try {
        // We need to support TS execution or compile first. 
        // Since we are in a TS project, running this directly with node might fail due to imports.
        // So let's write a TS file and run with tsx/ts-node.
        await initializeDatabase();
        console.log('Database initialized successfully.');
    } catch (error) {
        console.error('Failed to initialize database:', error);
        process.exit(1);
    }
}

main();
