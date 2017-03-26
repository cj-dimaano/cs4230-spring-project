// File: NNExperiment1.java
// Author: CJ Dimaano
// CS 5350 - Machine Learning
// Fall 2016
// Date created: December 5, 2016
// Last updated: December 5, 2016


package ml;


import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;


public class NNExperiment1 extends Experiment {

  public static void main(String[] args) {
    (new NNExperiment1()).run();
  }

  private double[][][] W;
  // private Set<Integer> remove = null;

  public double getPrediction(double[] x_i) {
    return (computeOutput(W, x_i) > 0 ? 1 : -1);
  }

  private double computeOutput(double[][][] w, double[] x_i) {
    double[] x = Arrays.copyOf(x_i, x_i.length);
    for(int l = 1; l < w.length; l++) {
      double[] z = new double[w[l - 1].length + 1];
      z[0] = 1;
      for(int j = 1; j < z.length; j++)
        z[j] = 1.0 / (1.0 + Math.exp(-MathEx.dot(w[l - 1][j - 1], x)));
      x = z;
    }
    return MathEx.dot(w[w.length - 1][0], x);
  }

  public void run() {

    if(!loadData(Data.TRAIN_SET))
      return;

    // remove = new HashSet<Integer>();
    // for(int i = 0; i < X.size(); i++) {
    //   double[] x_i = X.get(i);
    //   for(int j = 0; j < x_i.length; j++) {
    //     if(x_i[j] == 0)
    //       remove.add(j);
    //   }
    // }
    // for(int i = 0; i < X.size(); i++) {
    //   double[] x_i = X.get(i);
    //   double[] x_ip = new double[x_i.length - remove.size()];
    //   int ip = 0;
    //   for(int j = 0; j < x_i.length; j++) {
    //     if(!remove.contains(j)) {
    //       x_ip[ip] = x_i[j];
    //       ip++;
    //     }
    //   }
    //   X.set(i, x_ip);
    // }

    final int LAYERS = 1;
    // final int LAYER_NODES = X.size();
    final int LAYER_NODES = X.get(0).length / 2;
    // final int LAYER_NODES = (int)Math.log(X.get(0).length);
    // final int LAYER_NODES = (int)Math.log(X.size());
    // final int LAYER_NODES = 16;

    println("Layers: " + LAYERS);
    println("Layer nodes: " + LAYER_NODES);
    W = train(X, Y, LAYERS, LAYER_NODES, 100, 0.01);
    println("dataset\tacc\tpre\trec\tF1");
    StringBuilder sb = new StringBuilder();
    appendStringBuilder(sb, "train\t");
    test(X, Y, sb);
    println(sb.toString());

    if(!loadData(Data.TEST_SET))
      return;

    sb = new StringBuilder();
    appendStringBuilder(sb, "test\t");
    test(X, Y, sb);
    println(sb.toString());

    evaluate("nn2.csv");
    printerr("NNExperiment1 Complete.");
    
  }

  public boolean includeBias() {
    return true;
  }

  public double[] normalize(double[] x_i) {
    double[] xp = new double[x_i.length + Data.FEATURE_COUNT];
    for(int j = 1; j < x_i.length; j++) {
      xp[j] = 1.0 - Math.exp(-x_i[j]);
      xp[j + Data.FEATURE_COUNT] = 1.0 - Math.exp(-x_i[j] * x_i[j]);
    }
    return xp;
  }

  private double[][][] train(
    List<double[]> x,
    List<Double> y,
    int layers,
    int layerNodes,
    int epochs,
    double g0
  ) {

    double[][][] w = new double[layers + 1][][]; // + 1 for the output layer
    for(int l = 0; l < layers; l++) {
      w[l] = new double[layerNodes][];
      for(int z = 0; z < layerNodes; z++) {
        w[l][z] = new double[(l > 0 ? layerNodes + 1 : x.get(0).length)];
      }
    }
    w[layers] = new double[1][]; // Final layer has 1 node (the output node)
    w[layers][0] = new double[layerNodes + 1];
    reset(w);

    double[][][] w_swap = new double[w.length][][];
    for(int i = 0; i < w.length; i++) {
      w_swap[i] = new double[w[i].length][];
      for(int j = 0; j < w[i].length; j++)
        w_swap[i][j] = new double[w[i][j].length];
    }

    double[][] z = new double[w.length][];
    for(int l = 1; l < w.length; l++)
      z[l] = new double[w[l - 1].length + 1];
      
    for(int e = 0; e < epochs; e++) {
      Data.shuffle(x, y);
      for(int i = 0; i < x.size(); i++) {

        // Compute yp and remember hidden layer features.
        z[0] = Arrays.copyOf(x.get(i), x.get(i).length);
        for(int j = 0; j < z[0].length; j++)
          assert x.get(i)[j] == z[0][j];
        for(int l = 1; l < w.length; l++) {
          z[l][0] = 1;
          for(int j = 1; j < z[l].length; j++)
            z[l][j] = 1.0 / (1.0 + Math.exp(-MathEx.dot(w[l - 1][j - 1], z[l - 1])));
        }
        double yp = MathEx.dot(w[w.length - 1][0], z[z.length - 1]);

        // Save derivitive of square loss.
        double dLy = (yp - y.get(i));

        // Back propagation.

        int l = w.length - 1;
        for(int j = 0; j < w[l][0].length; j++) {
          w_swap[l][0][j] = w[l][0][j] - g0 * dLy * z[l][j];
          z[l][j] *= (1 - z[l][j]);
        }

        for(l = l - 1; l >= 0; l--) {
          for(int n = 0; n < w[l].length; n++) {
            for(int j = 0; j < w[l][n].length; j++)
              w_swap[l][n][j] = w[l][n][j] - g0 * dLy * propagate(w, z, l, n, j);
            for(int j = 0; j < w[l][n].length; j++)
              z[l][j] *= (1 - z[l][j]);
          }
        }
        double[][][] temp = w;
        w = w_swap;
        w_swap = temp;

        //
      }
    }

    return w;
  }

  private static double propagate(double[][][] w, double[][] z, int l, int n, int j) {
    double result = 0;

    if(l + 2 < w.length) {
      for(int n1 = 0; n1 < w[l + 1].length; n1++)
        result += z[l][j] * w[l + 1][n1][n + 1] * propagate(w, z, l + 1, n1, n + 1);
    }
    else
      result = z[l][j] * w[l + 1][0][n + 1] * z[l + 1][n + 1];

    return result;
  }

  private void reset(double[][][] w) {
    for(double[][] l : w)
      for(double[] n : l)
        for(int j = 0; j < n.length; j++)
          n[j] = Math.random() * 2 - 1;
          // n[j] = 1;
  }

  private void printWeights(double[][][] w) {
    for(double[][] l : w) {
      for(double[] n : l) {
        System.out.print("[");
        for(double j : n)
          System.out.print(" " + j);
        System.out.println(" ]");
      }
      System.out.println();
    }
  }

}