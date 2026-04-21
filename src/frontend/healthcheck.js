// Docker healthcheck for the Vite dev server.
// Executed inside the container: node /healthcheck.js
const http = require("http");

http
    .get(
        "http://localhost:" + (process.env.FRONTEND_PORT || 3030),
        function (res) {
            process.exit(res.statusCode < 400 ? 0 : 1);
        }
    )
    .on("error", function () {
        process.exit(1);
    });
