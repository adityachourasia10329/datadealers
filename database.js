import express from "express";
import sqlite3 from "sqlite3";

const app = express();
const db = new sqlite3.Database("app.db");

app.get("/users", (req, res) => {
  db.all("SELECT * FROM users", [], (err, rows) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json(rows);
  });
});

app.listen(3000);

import express from "express";
import sqlite3 from "sqlite3";

const app = express();
const db = new sqlite3.Database("app.db");

db.serialize(() => {
  db.run(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL
    )
  `);

  db.run("INSERT INTO users (name) VALUES (?)", ["Aditya"]);
});

app.get("/users", (req, res) => {
  db.all("SELECT * FROM users", [], (err, rows) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json(rows);
  });
});

app.listen(3000);