using System;
using System.Threading;

namespace Task05
{
  public class FineGrainedTree<T> where T : IComparable<T>
  {
    public class Node
    {
      public Mutex mutex;
      public T value;
      public Node left;
      public Node right;

      public Node(T value)
      {
        this.value = value;
        this.left  = null;
        this.right = null;
        this.mutex = new Mutex();
      }
    }
  
    private Mutex startMutex = new Mutex(); 
    private Node root = null;

    public bool find(T value)
    {
      startMutex.WaitOne();

      if (null == root)
      {
        startMutex.ReleaseMutex();
        return false;
      }
      else
      {
        root.mutex.WaitOne();
        Node buf1 = null;
        Node buf2 = root;

        while (null != buf2 && 0 != value.CompareTo(buf2.value))
        {
          if (null == buf1)
          {
            startMutex.ReleaseMutex();
          }
          else
          {
            buf1.mutex.ReleaseMutex();
          }

          buf1 = buf2;
          
          if (0 > value.CompareTo(buf2.value))
          {
            if (null != buf2.left)
            {
              buf2.left.mutex.WaitOne();
            }
            buf2 = buf2.left;
          }
          else
          {
            if (null != buf2.right)
            {
              buf2.right.mutex.WaitOne();
            }
            buf2 = buf2.right;            
          }
        }

        if (null == buf1)
        {
          startMutex.ReleaseMutex();
        }
        else
        {
          buf1.mutex.ReleaseMutex();
        }        

        if (null != buf2)
        {
          buf2.mutex.ReleaseMutex();
          return true;
        }
        else
        {
          return false;
        }
      }
    }
  
    public void insert(T value)
    {
      startMutex.WaitOne();

      if (null == root)
      {
        root = new Node(value);
        startMutex.ReleaseMutex();
      }
      else
      {
        root.mutex.WaitOne();

        Node buf1 = null;
        Node buf2 = root;

        while (null != buf2)
        {
          if (0 > value.CompareTo(buf2.value))
          {
            if (null != buf2.left)
            {
              buf2.left.mutex.WaitOne();
            }

            if (null == buf1)
            {
              startMutex.ReleaseMutex();
            }
            else
            {
              buf1.mutex.ReleaseMutex();
            }

            buf1 = buf2;
            buf2 = buf2.left;
          }
          else
          {
            if (null != buf2.right)
            {
              buf2.right.mutex.WaitOne();
            }

            if (null == buf1)
            {
              startMutex.ReleaseMutex();
            }
            else
            {
              buf1.mutex.ReleaseMutex();
            }

            buf1 = buf2;
            buf2 = buf2.right;
          }
        }

        buf2 = new Node(value);
        
        if (0 > value.CompareTo(buf1.value))
        {
          buf1.left = buf2;
        }
        else
        {
          buf1.right = buf2;
        }

        buf1.mutex.ReleaseMutex();
      }
    }
  
    public void remove(T value)
    {
      startMutex.WaitOne();

      if (null == root)
      {
        startMutex.ReleaseMutex();
        return;
      }
      else
      {
        root.mutex.WaitOne();
        Node buf1 = null;
        Node buf2 = root;

        while (null != buf2 && 0 != value.CompareTo(buf2.value))
        {
          if (0 > value.CompareTo(buf2.value))
          {
            if (null != buf2.left)
            {
              buf2.left.mutex.WaitOne();
            }

            if (root == buf1)
            {
              startMutex.ReleaseMutex();
            }
            else if (null != buf1)
            {
              buf1.mutex.ReleaseMutex();
            }

            buf1 = buf2;
            buf2 = buf2.left;
          }
          else
          {
            if (null != buf2.right)
            {
              buf2.right.mutex.WaitOne();
            }

            if (root == buf1)
            {
              startMutex.ReleaseMutex();
            }
            else if (null != buf1)
            {
              buf1.mutex.ReleaseMutex();
            }

            buf1 = buf2;
            buf2 = buf2.right;
          }
        }

        if (null == buf2)
        {
          buf1?.mutex.ReleaseMutex();

          if (root == buf1 || null == buf1)
          {
            startMutex.ReleaseMutex();
          }

          return;
        }
        else
        {
          if (null == buf2.left && null == buf2.right)
          {
            if (null == buf1)
            {
              root = null;
              buf2.mutex.ReleaseMutex();
              startMutex.ReleaseMutex();
            }
            else
            {
              if (0 > value.CompareTo(buf1.value))
              {
                buf1.left = null;
              }
              else
              {
                buf1.right = null;
              }

              buf2.mutex.ReleaseMutex();
              buf1.mutex.ReleaseMutex();

              if (root == buf1)
              {
                startMutex.ReleaseMutex();
              }
            }
          }
          else if (null != buf2.left && null == buf2.left.right)
          {
            buf2.left.mutex.WaitOne();
            
            if (null == buf1)
            {
              root = buf2.left;
              root.right = buf2.right;

              buf2.mutex.ReleaseMutex();
              root.mutex.ReleaseMutex();
              startMutex.ReleaseMutex();
            }
            else
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

              buf2.left.mutex.ReleaseMutex();
              buf2.mutex.ReleaseMutex();
              buf1.mutex.ReleaseMutex();

              if (root == buf1)
              {
                startMutex.ReleaseMutex();
              }
            }
          }
          else if (null == buf2.left && null != buf2.right)
          {
            buf2.right.mutex.WaitOne();
            
            if (null == buf1)
            {
              root = buf2.right;

              buf2.mutex.ReleaseMutex();
              root.mutex.ReleaseMutex();
              startMutex.ReleaseMutex();
            }
            else
            {
              if (0 > value.CompareTo(buf1.value))
              {
                buf1.left = buf2.right;
              }
              else
              {
                buf1.right = buf2.right;
              }

              buf2.right.mutex.ReleaseMutex();
              buf2.mutex.ReleaseMutex();
              buf1.mutex.ReleaseMutex();

              if (root == buf1)
              {
                startMutex.ReleaseMutex();
              }
            }
          }
          else
          {
            buf2?.left.mutex.WaitOne();
            Node buf3 = buf2.left;

            buf3?.right?.mutex.WaitOne();
            Node buf4 = buf3?.right;

            while (null != buf4?.right)
            {
              buf4.right.mutex.WaitOne();
              buf3.mutex.ReleaseMutex();
              buf3 = buf4;
              buf4 = buf4.right;
            }

            if (null == buf1)
            {
              root = buf4;
            }
            else
            {
              if (0 > value.CompareTo(buf1.value))
              {
                buf1.left = buf4;
              }
              else
              {
                buf1.right = buf4;
              }
            }

            buf3.right = null;
            buf4.left = buf2.left;
            buf4.right = buf2.right;

            buf4.mutex.ReleaseMutex();
            buf3.mutex.ReleaseMutex();
            buf2.mutex.ReleaseMutex();
            buf1?.mutex.ReleaseMutex();

            if (null == buf1 || root == buf1)
            {
              startMutex.ReleaseMutex();
            }
          }
        }
      }
    }
  }
}
