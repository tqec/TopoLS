//! Input-port layout and the lift of a finished layer (ports.py).

use indexmap::IndexMap;

use crate::geometry::Cell;
use crate::routing::color::Axis;

/// `auto_ports`: qubit i -> cell on a serpentine grid at `z_level`, `length`
/// per row, `edge_dist` apart. Orientation 'i' and type 0 for every port.
pub fn auto_ports(num_qubits: usize, z_level: i32, edge_dist: i32, length: Option<usize>) -> IndexMap<usize, Cell> {
    let length = length.unwrap_or_else(|| (num_qubits as f64).sqrt().ceil() as usize);
    let width = (num_qubits + length - 1) / length;
    let mut loc = IndexMap::new();
    let mut idx = 0usize;
    'rows: for j in 0..width {
        let row: Vec<usize> = if j % 2 == 0 { (0..length).collect() } else { (0..length).rev().collect() };
        for i in row {
            if idx >= num_qubits {
                break 'rows;
            }
            loc.insert(idx, Cell::new((i as i32) * edge_dist, (j as i32) * edge_dist, z_level));
            idx += 1;
        }
        if idx >= num_qubits {
            break;
        }
    }
    loc
}

pub const PORT_ORI: Axis = Axis::I;
