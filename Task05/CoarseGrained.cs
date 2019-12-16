using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace Task05
{
  public class Tree<T> where T : IComparable<T>
  {
    public class Node
    {
      public T value;
      public Node left;
      public Node right;

      public Node(T value)
      {
        this.value = value;
        this.left  = null;
        this.right = null;
      }
    }

    private Node root = null;

    virtual public bool find(T value)
    {
      Node buf = root;
      while (null != buf && 0 != value.CompareTo(buf.value))
      {
        if (0 > value.CompareTo(buf.value))
        {
          buf = buf.left;
        }
        else
        {
          buf = buf.right;
        }
      }

      if (null != buf)
      {
        return true;
      }
      else
      {
        return false;
      }
    }

    virtual public void insert(T value)
    {
      if (null == root)
      {
        root = new Node(value);
        return;
      }

      Node buf1 = null;
      Node buf2 = root;

      while (null != buf2)
      {
        if (0 > value.CompareTo(buf2.value))
        {
          buf1 = buf2;
          buf2 = buf2.left;
        }
        else
        {
          buf1 = buf2;
          buf2 = buf2.right;
        }
      }

      if (0 > value.CompareTo(buf1.value))
      {
        buf1.left = new Node(value);
      }
      else
      {
        buf1.right = new Node(value);
      }
    }

    virtual public void remove(T value)
    {
      Node buf1 = null;
      Node buf2 = root;

      while (null != buf2 && 0 != value.CompareTo(buf2.value))
      {
        if (0 > value.CompareTo(buf2.value))
        {
          buf1 = buf2;
          buf2 = buf2.left;
        }
        else
        {
          buf1 = buf2;
          buf2 = buf2.right;
        }
      }

      if (null == buf2)
      {
        return;
      }
      else if (null == buf1)
      {
        if (null == buf2.left)
        {
          root = buf2.right;
        }
        else if (null == buf2.left.right)
        {
          root = buf2.left;
          root.right = buf2.right;
        }
        else
        {
          Node buf3 = buf2.left;
          Node buf4 = buf3.right;

          while (null != buf4.right)
          {
            buf3 = buf4;
            buf4 = buf4.right;
          }

          buf3.right = null;
          root       = buf4;
          root.left  = buf2.left;
          root.right = buf2.right;
          
        }
      }
      else
      {
        if (null == buf2.left)
        {
          if (0 > value.CompareTo(buf1.value))
          {
            buf1.left = buf2.right;
          }
          else
          {
            buf1.right = buf2.right;
          }
        }
        else if (null == buf2.right)
        {
          if (0 > value.CompareTo(buf1.value))
          {
            buf1.left = buf2.left;
          }
          else
          {
            buf1.right = buf2.left;
          }
        }
        else if (null == buf2.left.right)
        {
          if (0 > value.CompareTo(buf1.value))
          {
            buf1.left = buf2.left;
            buf1.left.right = buf2.right;
          }
          else
          {
            buf1.right = buf2.left;
            buf1.right.right = buf2.right;
          }
        }
        else
        {
          Node buf3 = buf2.left;
          Node buf4 = buf3.right;

          while (null != buf4.right)
          {
            buf3 = buf4;
            buf4 = buf4.right;
          }

          if (0 > value.CompareTo(buf1.value))
          {
            buf1.left = buf4;
            buf1.left.right = buf2.right;
          }
          else
          {
            buf1.right = buf4;
            buf1.right.right = buf2.right;
          }

          buf3.right = null;
        }
      }
    }
  }

  public class CoarseGrainedTree<T> : Tree<T> where T : IComparable<T>
  {
      private Mutex mutex = new Mutex();
      
      override public bool find(T value)
      {
        mutex.WaitOne();

        bool res = base.find(value);

        mutex.ReleaseMutex();

        return res;
      }

      override public void insert(T value)
      {
        mutex.WaitOne();

        base.insert(value);

        mutex.ReleaseMutex();
      }

      override public void remove(T value)
      {
        mutex.WaitOne();

        base.remove(value);

        mutex.ReleaseMutex();
      }
  }
}
