import javax.swing.*;
import java.awt.*;
import java.sql.*;

public class AdminTrainManagement extends JFrame {
    private JTextField trainNameField;

    public AdminTrainManagement() {
        setTitle("Admin - Train Management");
        setSize(400, 200);
        setLayout(new GridLayout(3, 2));

        add(new JLabel("Train Name:"));
        trainNameField = new JTextField();
        add(trainNameField);

        JButton addBtn = new JButton("Add Train");
        addBtn.addActionListener(e -> addTrain());
        add(addBtn);

        JButton deleteBtn = new JButton("Delete Train");
        deleteBtn.addActionListener(e -> deleteTrain());
        add(deleteBtn);

        setVisible(true);
    }

    private void addTrain() {
        String trainName = trainNameField.getText();

        try (Connection conn = DatabaseConnection.getConnection()) {
            String sql = "INSERT INTO trains (train_name) VALUES (?)";
            PreparedStatement stmt = conn.prepareStatement(sql);
            stmt.setString(1, trainName);
            stmt.executeUpdate();

            JOptionPane.showMessageDialog(this, "Train added successfully!");
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    private void deleteTrain() {
        String trainName = trainNameField.getText();

        try (Connection conn = DatabaseConnection.getConnection()) {
            String sql = "DELETE FROM trains WHERE train_name=?";
            PreparedStatement stmt = conn.prepareStatement(sql);
            stmt.setString(1, trainName);
            int rows = stmt.executeUpdate();

            if (rows > 0) {
                JOptionPane.showMessageDialog(this, "Train deleted successfully!");
            } else {
                JOptionPane.showMessageDialog(this, "Train not found!");
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
