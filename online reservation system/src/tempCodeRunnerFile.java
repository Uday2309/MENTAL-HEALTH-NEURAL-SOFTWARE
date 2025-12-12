import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

public class DatabaseConnection {
    private static final String DB_URL = "jdbc:mysql://localhost:3306/reservation_system";
    private static final String USER = "uday"; // your MySQL username
    private static final String PASSWORD = "23092005@Uday"; // your MySQL password

    static {
        try (Connection conn = getConnection(); Statement stmt = conn.createStatement()) {
            // Create trains table
            stmt.execute("CREATE TABLE IF NOT EXISTS trains (" +
                    "train_id INT AUTO_INCREMENT PRIMARY KEY," +
                    "train_name VARCHAR(100) NOT NULL," +
                    "source VARCHAR(100) NOT NULL," +
                    "destination VARCHAR(100) NOT NULL," +
                    "available_seats INT NOT NULL)");

            // Create bookings table
            stmt.execute("CREATE TABLE IF NOT EXISTS bookings (" +
                    "pnr VARCHAR(50) PRIMARY KEY," +
                    "train_id INT," +
                    "passenger_name VARCHAR(100) NOT NULL," +
                    "age INT," +
                    "FOREIGN KEY(train_id) REFERENCES trains(train_id))");

            // Create users table
            stmt.execute("CREATE TABLE IF NOT EXISTS users (" +
                    "username VARCHAR(50) PRIMARY KEY," +
                    "password VARCHAR(100) NOT NULL," +
                    "role VARCHAR(50) NOT NULL)");

            System.out.println("Database initialized successfully!");
        } catch (SQLException e) {
            System.out.println("Database initialization failed: " + e.getMessage());
        }
    }

    public static Connection getConnection() throws SQLException {
        return DriverManager.getConnection(DB_URL, USER, PASSWORD);
    }
}
