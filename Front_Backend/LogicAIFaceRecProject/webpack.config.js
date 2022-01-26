const path = require('path')

module.exports = {
    entry: './js/camera.js',
    
    output: {
        filename: 'camera.js',
        path: path.resolve(__dirname, 'dist')
    },
    devServer: {
        port: 9000
    }
    // resolve: {
    //     // ... rest of the resolve config
    //     fallback: {
    //       "fs": false,
    //       "buffer": require.resolve("buffer"),
    //       "path": require.resolve("path-browserify"),
    //       "assert": require.resolve("assert"),
    //       "stream": require.resolve("stream"),
    //       "url": require.resolve("url"),
    //     }
    //   },
}