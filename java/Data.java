// File: Data.java
// Author: CJ Dimaano
// CS 5350 - Machine Learning
// Fall 2016
// Date created: November 14, 2016
// Last updated: December 13, 2016


package ml;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;


public class Data {
  
  public static final int FEATURE_COUNT = 360;
  public static final String TRAIN_SET = "data/data-splits/data.train";
  public static final String TEST_SET = "data/data-splits/data.test";

  public static boolean loadData(
    String file,
    List<double[]> x, 
    List<Double> y,
    boolean includeBias
  ) {
    x.clear();
    y.clear();
    FileScanner fs = new FileScanner() {
      public boolean parseLine(String line) {
        double y_i = 0;
        double[] x_i = new double[(includeBias ? FEATURE_COUNT + 1 : FEATURE_COUNT)];
        Scanner s = new Scanner(line);
        y_i = (s.nextInt() > 0 ? 1 : -1);
        if(includeBias)
          x_i[0] = 1;
        while(s.hasNext()) {
          String[] feature = s.next().split(":");
          int f = Integer.parseInt(feature[0]);
          x_i[(includeBias ? f : f - 1)] = Double.parseDouble(feature[1]);
        }
        s.close();
        y.add(y_i);
        x.add(x_i);
        return true;
      }
    };
    return fs.scanByLine(file);
  }


  public static Set<Integer> pruneFeatures(List<double[]> x, List<Double> y) {
    Set<Integer> result = new HashSet<Integer>();
    for(int i = 0; i < x.size(); i++) {
      if(y.get(i) > 0) {
        double[] x_i = x.get(i);
        for(int j = 0; j < x_i.length; j++) {
          if(x_i[j] > 0)
            result.add(j);
        }
      }
    }
    return result;
  }

  public static Set<Integer> fullPruneFeatures(List<double[]> x, List<Double> y) {
    Set<Integer> result = new HashSet<Integer>();
    for(int j = 0; j < x.get(0).length; j++)
      result.add(j);
    for(int i = 0; i < x.size(); i++) {
      if(y.get(i) > 0) {
        double[] x_i = x.get(i);
        for(int j = 0; j < x_i.length; j++)
          if(x_i[j] == 0)
            result.remove(j);
      }
    }
    return result;
  }

  /**
   * shuffle
   *   Shuffles a data set.
   */
  public static <T1, T2> void shuffle(List<T1> l1, List<T2> l2) {
    for(int i = 0; i < l1.size(); i++) {
      int j = (int)(Math.random() * l1.size());
      swap(i, j, l1);
      swap(i, j, l2);
    }
  }

  /**
   * swap
   *   Swaps two values in a list.
   */
  public static <T> void swap(int i, int j, List<T> l) {
    T temp = l.get(i);
    l.set(i, l.get(j));
    l.set(j, temp);
  }

  /**
   * sortExamples
   *   Sorts a set of examples by label.
   */
  public static void sortExamples(
    List<double[]> x,
    List<Double> y
  ) {
    List<double[]> u = new ArrayList<double[]>(x);
    List<Double> v = new ArrayList<Double>(y);
    x.clear();
    y.clear();

    for(int i = 0; i < u.size(); i++) {
      if(v.get(i) < 0) {
        x.add(u.get(i));
        y.add(v.get(i));
      }
    }
    for(int i = 0; i < u.size(); i++) {
      if(v.get(i) > 0) {
        x.add(u.get(i));
        y.add(v.get(i));
      }
    }
  }

  /**
   * foldExamples
   *   Splits a data set into training examples and test examples.
   */
  public static void foldExamples(
    List<double[]> x,
    List<Double> y,
    int folds,
    int fold,
    List<double[]> x_train,
    List<Double> y_train,
    List<double[]> x_test,
    List<Double> y_test
  ) {
    x_train.clear();
    y_train.clear();
    x_test.clear();
    y_test.clear();
    for(int i = 0; i < x.size(); i++) {
      if(i % folds == fold) {
        x_test.add(x.get(i));
        y_test.add(y.get(i));
      }
      else {
        x_train.add(x.get(i));
        y_train.add(y.get(i));
      }
    }
  }

  /**
   * filterExamples
   *   Gets the subset of examples where the given feature has a given value.
   */
  public static void filterExamples(
    List<double[]> x,
    List<Double> y,
    List<double[]> x_v,
    List<Double> y_v,
    int f,
    double v
  ) {
    x_v.clear();
    y_v.clear();
    for(int i = 0; i < x.size(); i++) {
      double[] x_i = x.get(i);
      if(x_i[f] == v) {
        x_v.add(x_i);
        y_v.add(y.get(i));
      }
    }
  }

  /**
   * splitExamples
   *   Splits a set of examples into a set with the value of a given feature
   *   below a given threshold, and a set with the value of a given feature
   *   above or equal to the given threshold.
   */
  public static void splitExamples(
    List<double[]> x,
    List<Double> y,
    List<double[]> x_above,
    List<Double> y_above,
    List<double[]> x_below,
    List<Double> y_below,
    int f,
    double v
  ) {
    x_above.clear();
    y_above.clear();
    x_below.clear();
    y_below.clear();
    for(int i = 0; i < x.size(); i++) {
      double[] x_i = x.get(i);
      if(x_i[f] < v) {
        x_below.add(x_i);
        y_below.add(y.get(i));
      }
      else {
        x_above.add(x_i);
        y_above.add(y.get(i));
      }
    }
  }

  /**
   * getLabelCounts
   *   Gets the counts for all label values that occur in a set of examples.
   */
  public static Map<Double, Integer> getLabelCounts(
    List<Double> y
  ) {
    Map<Double, Integer> result = new HashMap<Double, Integer>();
    for(Double y_i : y) {
      if(!result.containsKey(y_i))
        result.put(y_i, 0);
      result.put(y_i, result.get(y_i) + 1);
    }
    return result;
  }

  /**
   * max
   *   Finds the key with the highest value from a table of values.
   *
   * <p>
   *   Ties are broken by comparing the keys. The "lesser" key trumps the
   *   "greater" key.
   * </p>
   *
   *
   * @param values
   *   The table of values over which to search.
   *
   *
   * @return
   *   The key whose value is the greatest out of all the other values in the
   *   given table.
   */
  public static <K extends Comparable<K>, V extends Comparable<V>> K max(
    Map<K, V> values
  ) {
    K result = null;
    for(K key : values.keySet()) {
      if(result == null)
        result = key;
      else {
        int cmp = values.get(result).compareTo(values.get(key));
        if(cmp < 0)
          result = key;
        else if(cmp == 0 && result.compareTo(key) < 0)
          result = key;
      }
    }
    return result;
  }

  /**
   * entropy
   *   Computes the entropy for a given set of labels.
   */
  public static double entropy(List<Double> y) {
    int pos = 0;
    for(Double y_i : y)
      if(y_i > 0)
        pos++;
    double p = (double)pos / (double)y.size();
    double q = 1 - p;
    return -p * MathEx.log2(p) - q * MathEx.log2(q);
  }

}