using Microsoft.VisualStudio.TestTools.UnitTesting;
using Task05;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace Task05Test
{
  [TestClass]
  public class CoarseGrainedTreeTests
  {
    [TestMethod]
    public void SearchInEmptyCGTree()
    {
      var cgtree = new CoarseGrainedTree<int>();

      Assert.AreEqual<bool>(false, cgtree.find(0));
    }

    [TestMethod]
    public void AddSingleRootToCGTree()
    {
        var cgtree = new CoarseGrainedTree<int>();

        cgtree.insert(0);

        Assert.AreEqual<bool>(true, cgtree.find(0));
    }

    [TestMethod]
    public void AddForkToCGTree()
    {
        var cgtree = new CoarseGrainedTree<int>();

        cgtree.insert(0);
        cgtree.insert(-1);
        cgtree.insert(1);      

        Assert.AreEqual<bool>(true, cgtree.find(0));
        Assert.AreEqual<bool>(true, cgtree.find(-1));
        Assert.AreEqual<bool>(true, cgtree.find(1));
    }

    [TestMethod]
    public void AddSequenceToCGTree()
    {
      var cgtree = new CoarseGrainedTree<int>();

      cgtree.insert(0);
      cgtree.insert(1);
      cgtree.insert(2);   

      Assert.AreEqual<bool>(true, cgtree.find(0));
      Assert.AreEqual<bool>(true, cgtree.find(1));
      Assert.AreEqual<bool>(true, cgtree.find(2));
    }

    [TestMethod]
    public void SequentialAddAndRemoveInCGTree()
    {
      var cgtree = new CoarseGrainedTree<int>();

      var wholeArray  = new List<int>(){4, 2, 1, 3, 6, 5, 7};
      var partRemove = new List<int>(){4, 7, 2};
      var others     = new List<int>(){1, 3, 5, 6};

      foreach (var item in wholeArray)
      {
        cgtree.insert(item);
      }

      foreach (var item in wholeArray)
      {
        Assert.AreEqual<bool>(true, cgtree.find(item));
      }

      foreach (var item in partRemove)
      {
        cgtree.remove(item);
      }

      foreach (var item in partRemove)
      {
        Assert.AreEqual<bool>(false, cgtree.find(item));
      }

      foreach (var item in others)
      {
        Assert.AreEqual<bool>(true, cgtree.find(item));
      }
    }

    [TestMethod]
    public void ParallelAddAndRemoveInCGTree()
    {
      var cgtree = new CoarseGrainedTree<int>();

      var wholeArray  = new int[] {4, 2, 1, 3, 6, 5, 7, 0};
      var partRemove  = new int[] {4, 7, 2, 0};
      var others      = new int[] {1, 3, 5, 6};

      Task[] tasks1 = new Task[8];
      Task[] tasks2 = new Task[4];

      Parallel.For(0, wholeArray.Length, index => 
      {
        tasks1[index] = Task.Run( () => {cgtree.insert(wholeArray[index]);});
      });

      Task.WaitAll(tasks1);

      Parallel.For(0, wholeArray.Length, index => 
      {
        tasks1[index] = Task.Run( () => {Assert.AreEqual<bool>(true, cgtree.find(wholeArray[index]));});
      });

      Task.WaitAll(tasks1);

      Parallel.For(0, partRemove.Length, index => 
      {
        tasks2[index] = Task.Run( () => {cgtree.remove(partRemove[index]);});
      });

      Task.WaitAll(tasks2);

      Parallel.For(0, partRemove.Length, index => 
      {
        tasks2[index] = Task.Run( () => {Assert.AreEqual<bool>(false, cgtree.find(partRemove[index]));});
      });

      Task.WaitAll(tasks2);

      Parallel.For(0, others.Length, index => 
      {
        tasks2[index] = Task.Run( () => {Assert.AreEqual<bool>(true, cgtree.find(others[index]));});
      });

      Task.WaitAll(tasks2);
    }
  }
}
