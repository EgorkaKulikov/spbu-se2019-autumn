using System.Collections.Generic;
using System.Threading;

namespace Task03 {
    public abstract class AccessPolicy {
        public abstract void acquire_read();
        public abstract void release_read();
        public abstract void acquire_write();
        public abstract void release_write();
    }

    public class Storage<T> {

        public interface Reader {
            void doReading(T data);
        }

        public interface Writer {
            void doWriting(T data);
        }

        public void add_reader(Reader reader) {
            policy.acquire_read();
            reader.doReading(data);
            policy.release_read();
        }

        public void add_writer(Writer writer) {
            policy.acquire_write();
            writer.doWriting(data);
            policy.release_write();
        }

        private T data;
        private AccessPolicy policy;

        public Storage(AccessPolicy policy, T data) {
            this.policy = policy;
            this.data = data;
        }
    }
}
