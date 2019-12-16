using Microsoft.VisualStudio.TestTools.UnitTesting;
using Task05;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace Task05Test
{
  [TestClass]
  public class FineGrainedTreeTests
  {
    [TestMethod]
    public void SearchInEmptyFGTree()
    {
      var fgtree = new FineGrainedTree<int>();

      Assert.AreEqual<bool>(false, fgtree.find(0));
    }

    [TestMethod]
    public void AddSingleRootToFGTree()
    {
        var fgtree = new FineGrainedTree<int>();

        fgtree.insert(0);

        Assert.AreEqual<bool>(true, fgtree.find(0));
    }

    [TestMethod]
    public void AddForkToFGTree()
    {
        var fgtree = new FineGrainedTree<int>();

        fgtree.insert(0);
        fgtree.insert(-1);
        fgtree.insert(1);      

        Assert.AreEqual<bool>(true, fgtree.find(0));
        Assert.AreEqual<bool>(true, fgtree.find(-1));
        Assert.AreEqual<bool>(true, fgtree.find(1));
    }

    [TestMethod]
    public void AddSequenceToFGTree()
    {
      var fgtree = new FineGrainedTree<int>();

      fgtree.insert(0);
      fgtree.insert(1);
      fgtree.insert(2);   

      Assert.AreEqual<bool>(true, fgtree.find(0));
      Assert.AreEqual<bool>(true, fgtree.find(1));
      Assert.AreEqual<bool>(true, fgtree.find(2));
    }

    [TestMethod]
    public void SequentialAddAndRemoveInFGTree()
    {
      var fgtree = new FineGrainedTree<int>();

      var wholeArray  = new List<int>(){4, 2, 1, 3, 6, 5, 7};
      var partRemove = new List<int>(){4, 7, 2};
      var others     = new List<int>(){1, 3, 5, 6};

      foreach (var item in wholeArray)
      {
        fgtree.insert(item);
      }

      foreach (var item in wholeArray)
      {
        Assert.AreEqual<bool>(true, fgtree.find(item));
      }

      foreach (var item in partRemove)
      {
        fgtree.remove(item);
      }

      foreach (var item in partRemove)
      {
        Assert.AreEqual<bool>(false, fgtree.find(item));
      }

      foreach (var item in others)
      {
        Assert.AreEqual<bool>(true, fgtree.find(item));
      }
    }

    [TestMethod]
    public void ParallelAddAndRemoveInFGTree()
    {
      var fgtree = new FineGrainedTree<int>();

      var wholeArray = new int[] {4, 2, 1, 3, 6, 5, 7, 0};
      var partRemove = new int[] {4, 7, 2, 0};
      var others     = new int[] {1, 3, 5, 6};

      var tasks1 = new Task[8];
      var tasks2 = new Task[4];

      Parallel.For(0, wholeArray.Length, index => 
      {
        tasks1[index] = Task.Run( () => {fgtree.insert(wholeArray[index]);});
      });

      Task.WaitAll(tasks1);

      Parallel.For(0, wholeArray.Length, index => 
      {
        tasks1[index] = Task.Run( () => {Assert.AreEqual<bool>(true, fgtree.find(wholeArray[index]));});
      });

      Task.WaitAll(tasks1);

      Parallel.For(0, partRemove.Length, index => 
      {
        tasks2[index] = Task.Run( () => {fgtree.remove(partRemove[index]);});
      });

      Task.WaitAll(tasks2);

      Parallel.For(0, partRemove.Length, index => 
      {
        tasks2[index] = Task.Run( () => {Assert.AreEqual<bool>(false, fgtree.find(partRemove[index]));});
      });

      Task.WaitAll(tasks2);

      Parallel.For(0, others.Length, index => 
      {
        tasks2[index] = Task.Run( () => {Assert.AreEqual<bool>(true, fgtree.find(others[index]));});
      });

      Task.WaitAll(tasks2);
    }
  }
}
